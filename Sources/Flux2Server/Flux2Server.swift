import Foundation
import Network
import MLX
import Flux2Core
import FluxTextEncoders
import ImageIO
import UniformTypeIdentifiers
import CoreGraphics

extension MLXArray: @unchecked Sendable {}

struct LoRARequest: Codable, Hashable {
    var filePath: String
    var scale: Float?
}

struct GenerateRequest: Codable {
    var prompt: String
    var imagePath: String
    var outputPath: String
    var width: Int?
    var height: Int?
    var steps: Int?
    var guidance: Float?
    var seed: UInt64?
    var jobId: String?
    var loras: [LoRARequest]?
}

struct ProgressRecord: Codable {
    var jobId: String
    var currentStep: Int
    var totalSteps: Int
    var overallPercent: Int
    var currentStepPercent: Int
    var phase: String
}

final class ProgressStore: @unchecked Sendable {
    private let lock = NSLock()
    private var records: [String: ProgressRecord] = [:]

    func update(jobId: String, currentStep: Int, totalSteps: Int, phase: String = "denoise") {
        lock.lock(); defer { lock.unlock() }
        let total = max(1, totalSteps)
        let current = max(0, min(currentStep, total))
        records[jobId] = ProgressRecord(
            jobId: jobId,
            currentStep: current,
            totalSteps: total,
            overallPercent: Int((Double(current) / Double(total) * 100.0).rounded()),
            currentStepPercent: current == 0 ? 0 : 100,
            phase: phase
        )
    }

    func get(jobId: String) -> ProgressRecord? {
        lock.lock(); defer { lock.unlock() }
        return records[jobId]
    }

    func clear(jobId: String) {
        lock.lock(); defer { lock.unlock() }
        records.removeValue(forKey: jobId)
    }
}

struct GenerateResponse: Codable {
    var ok: Bool
    var outputPath: String?
    var elapsedSeconds: Double?
    var error: String?
}

func loadImage(from path: String) -> CGImage? {
    let url = URL(fileURLWithPath: path)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
    return image
}

func cropResizeImage(_ image: CGImage, width targetWidth: Int, height targetHeight: Int) -> CGImage? {
    guard targetWidth > 0, targetHeight > 0 else { return image }
    let srcW = image.width
    let srcH = image.height
    let targetAspect = CGFloat(targetWidth) / CGFloat(targetHeight)
    let srcAspect = CGFloat(srcW) / CGFloat(srcH)
    let cropRect: CGRect
    if srcAspect > targetAspect {
        let cropW = CGFloat(srcH) * targetAspect
        cropRect = CGRect(x: (CGFloat(srcW) - cropW) / 2.0, y: 0, width: cropW, height: CGFloat(srcH))
    } else {
        let cropH = CGFloat(srcW) / targetAspect
        // Slightly top-biased crop preserves faces/hats in selfie edits.
        let y = max(0, (CGFloat(srcH) - cropH) * 0.18)
        cropRect = CGRect(x: 0, y: y, width: CGFloat(srcW), height: cropH)
    }
    guard let cropped = image.cropping(to: cropRect.integral) else { return nil }
    guard let ctx = CGContext(
        data: nil,
        width: targetWidth,
        height: targetHeight,
        bitsPerComponent: 8,
        bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }
    ctx.interpolationQuality = .high
    ctx.draw(cropped, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))
    return ctx.makeImage()
}

func saveImage(_ image: CGImage, to path: String) throws {
    let url = URL(fileURLWithPath: path)
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    let utType: CFString = path.hasSuffix(".png") ? UTType.png.identifier as CFString : UTType.jpeg.identifier as CFString
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, utType, 1, nil) else {
        throw Flux2Error.imageProcessingFailed("Failed to create image destination")
    }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else {
        throw Flux2Error.imageProcessingFailed("Failed to write image")
    }
}

actor Flux2Service {
    let basePipeline: Flux2Pipeline
    let progressStore: ProgressStore
    let model: Flux2Model
    let quantization: Flux2QuantizationConfig
    let memoryOptimization: MemoryOptimizationConfig?
    let vaeVariant: ModelRegistry.VAEVariant
    let hfToken: String?
    let clearCacheEveryNSteps: Int
    let loraPipelineCacheLimit: Int
    var loraPipelines: [String: Flux2Pipeline] = [:]
    var loraPipelineOrder: [String] = []
    var embeddingCache: [String: MLXArray] = [:]

    init(progressStore: ProgressStore) {
        let env = ProcessInfo.processInfo.environment
        let textQuant = MistralQuantization(rawValue: env["FLUX2_TEXT_QUANT"] ?? "8bit") ?? .mlx8bit
        // Default to qint8: this is Flux2Core's balanced fast path and avoids the
        // previous bf16 transformer bottleneck. ComfyUI still controls steps.
        let transformerQuant = TransformerQuantization(rawValue: env["FLUX2_TRANSFORMER_QUANT"] ?? "qint8") ?? .qint8
        let modelName = (env["FLUX2_MODEL"] ?? "klein9BKV").lowercased()
        let model: Flux2Model = {
            switch modelName {
            case "klein9bkv", "klein-9b-kv", "kv": return .klein9BKV
            case "klein9bbase", "klein-9b-base", "base": return .klein9BBase
            case "klein4b", "klein-4b": return .klein4B
            default: return .klein9B
            }
        }()
        let quant = Flux2QuantizationConfig(textEncoder: textQuant, transformer: transformerQuant)
        let token = env["HF_TOKEN"]
        let memoryOptimization = Self.memoryOptimization(from: env)
        let vaeVariant = ModelRegistry.VAEVariant(rawValue: env["FLUX2_VAE_VARIANT"] ?? "standard") ?? .standard
        let clearCacheEveryNSteps = Int(env["FLUX2_CLEAR_CACHE_EVERY_N_STEPS"] ?? "0") ?? 0
        self.model = model
        self.quantization = quant
        self.memoryOptimization = memoryOptimization
        self.vaeVariant = vaeVariant
        self.hfToken = token
        self.clearCacheEveryNSteps = clearCacheEveryNSteps
        self.loraPipelineCacheLimit = max(1, Int(env["FLUX2_LORA_PIPELINE_CACHE_LIMIT"] ?? "1") ?? 1)
        self.basePipeline = Self.makePipeline(
            model: model,
            quantization: quant,
            memoryOptimization: memoryOptimization,
            vaeVariant: vaeVariant,
            hfToken: token,
            clearCacheEveryNSteps: clearCacheEveryNSteps
        )
        self.progressStore = progressStore
        if Self.envFlag(env["FLUX2_PROFILE"]) {
            Flux2Profiler.shared.enable()
        } else {
            Flux2Profiler.shared.disable()
        }
        Flux2Debug.setNormalMode()
        FluxDebug.isEnabled = false
        print("[Flux2Server] config model=\(model) textQuant=\(textQuant.rawValue) transformerQuant=\(transformerQuant.rawValue) memoryOptimization=\(memoryOptimization) vae=\(vaeVariant.rawValue) profile=\(Flux2Profiler.shared.isEnabled) loraCacheLimit=\(self.loraPipelineCacheLimit)")
        fflush(stdout)
    }

    private static func makePipeline(
        model: Flux2Model,
        quantization: Flux2QuantizationConfig,
        memoryOptimization: MemoryOptimizationConfig?,
        vaeVariant: ModelRegistry.VAEVariant,
        hfToken: String?,
        clearCacheEveryNSteps: Int
    ) -> Flux2Pipeline {
        let pipeline = Flux2Pipeline(
            model: model,
            quantization: quantization,
            memoryOptimization: memoryOptimization,
            vaeVariant: vaeVariant,
            hfToken: hfToken
        )
        pipeline.memoryProfile = .performance
        pipeline.clearCacheEveryNSteps = clearCacheEveryNSteps
        return pipeline
    }

    private static func envFlag(_ value: String?) -> Bool {
        guard let value else { return false }
        switch value.lowercased() {
        case "1", "true", "yes", "on": return true
        default: return false
        }
    }

    private static func memoryOptimization(from env: [String: String]) -> MemoryOptimizationConfig {
        switch (env["FLUX2_MEMORY_OPTIMIZATION"] ?? "disabled").lowercased() {
        case "auto":
            return MemoryOptimizationConfig.recommended(forRAMGB: Flux2MemoryManager.shared.physicalMemoryGB)
        case "light":
            return .light
        case "moderate":
            return .moderate
        case "aggressive":
            return .aggressive
        case "ultralow", "ultra-low", "ultra_low":
            return .ultraLowMemory
        default:
            return .disabled
        }
    }

    private func normalizedLoRAs(_ loras: [LoRARequest]?) throws -> [LoRARequest] {
        let fileManager = FileManager.default
        return try (loras ?? [])
            .filter { !$0.filePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .map { lora in
                let url = URL(fileURLWithPath: lora.filePath).standardizedFileURL
                guard fileManager.fileExists(atPath: url.path) else {
                    throw Flux2Error.invalidConfiguration("LoRA file does not exist: \(lora.filePath)")
                }
                return LoRARequest(filePath: url.path, scale: lora.scale ?? 1.0)
            }
            .sorted {
                if $0.filePath == $1.filePath {
                    return ($0.scale ?? 1.0) < ($1.scale ?? 1.0)
                }
                return $0.filePath < $1.filePath
            }
    }

    private func loraSignature(_ loras: [LoRARequest]) -> String {
        loras
            .map { "\($0.filePath)@\($0.scale ?? 1.0)" }
            .joined(separator: "|")
    }

    private func touchLoRAPipeline(signature: String) {
        loraPipelineOrder.removeAll { $0 == signature }
        loraPipelineOrder.append(signature)
    }

    private func pipeline(for loras: [LoRARequest]?) throws -> (pipeline: Flux2Pipeline, loraCount: Int, cacheHit: Bool) {
        let normalized = try normalizedLoRAs(loras)
        guard !normalized.isEmpty else {
            return (basePipeline, 0, true)
        }

        let signature = loraSignature(normalized)
        if let cached = loraPipelines[signature] {
            touchLoRAPipeline(signature: signature)
            return (cached, normalized.count, true)
        }

        let pipeline = Self.makePipeline(
            model: model,
            quantization: quantization,
            memoryOptimization: memoryOptimization,
            vaeVariant: vaeVariant,
            hfToken: hfToken,
            clearCacheEveryNSteps: clearCacheEveryNSteps
        )
        for lora in normalized {
            _ = try pipeline.loadLoRA(LoRAConfig(filePath: lora.filePath, scale: lora.scale ?? 1.0))
        }
        loraPipelines[signature] = pipeline
        touchLoRAPipeline(signature: signature)
        while loraPipelines.count > loraPipelineCacheLimit, let evict = loraPipelineOrder.first {
            loraPipelineOrder.removeFirst()
            loraPipelines.removeValue(forKey: evict)
            MLX.Memory.clearCache()
        }
        return (pipeline, normalized.count, false)
    }

    func generate(_ req: GenerateRequest) async -> GenerateResponse {
        let t0 = Date()
        if Flux2Profiler.shared.isEnabled {
            Flux2Profiler.shared.reset()
        }
        do {
            guard let loadedRef = loadImage(from: req.imagePath) else {
                throw Flux2Error.imageProcessingFailed("Failed to load reference image: \(req.imagePath)")
            }
            let targetWidth = req.width ?? loadedRef.width
            let targetHeight = req.height ?? loadedRef.height
            guard let ref = cropResizeImage(loadedRef, width: targetWidth, height: targetHeight) else {
                throw Flux2Error.imageProcessingFailed("Failed to prepare reference image")
            }
            let active = try pipeline(for: req.loras)
            let embeddings: MLXArray
            if let cached = embeddingCache[req.prompt] {
                embeddings = cached
            } else {
                embeddings = try await basePipeline.precomputeTextEmbeddings(prompt: req.prompt, upsamplePrompt: false)
                embeddingCache[req.prompt] = embeddings
            }
            let jobId = req.jobId ?? UUID().uuidString
            progressStore.update(jobId: jobId, currentStep: 0, totalSteps: req.steps ?? 1, phase: "starting")
            active.pipeline.resetTransformerInferenceCaches()
            let image = try await active.pipeline.generate(
                mode: .imageToImage(images: [ref]),
                prompt: req.prompt,
                interpretImagePaths: nil,
                height: targetHeight,
                width: targetWidth,
                steps: req.steps ?? 1,
                guidance: req.guidance ?? 1.0,
                seed: req.seed,
                upsamplePrompt: false,
                precomputedEmbeddings: embeddings,
                checkpointInterval: nil,
                onProgress: { [progressStore] current, total in
                    progressStore.update(jobId: jobId, currentStep: current, totalSteps: total)
                },
                onCheckpoint: nil,
                onStep: nil
            )
            try saveImage(image, to: req.outputPath)
            let elapsed = Date().timeIntervalSince(t0)
            print("[Flux2Server] job=\(jobId) width=\(targetWidth) height=\(targetHeight) steps=\(req.steps ?? 1) loras=\(active.loraCount) loraCacheHit=\(active.cacheHit) elapsed=\(String(format: "%.2f", elapsed))")
            if Flux2Profiler.shared.isEnabled {
                let report = Flux2Profiler.shared.generateReport()
                if !report.isEmpty {
                    print(report)
                }
            }
            fflush(stdout)
            return GenerateResponse(ok: true, outputPath: req.outputPath, elapsedSeconds: elapsed, error: nil)
        } catch {
            let message = String(describing: error)
            print("[Flux2Server] generate error: \(message)")
            fflush(stdout)
            return GenerateResponse(ok: false, outputPath: nil, elapsedSeconds: Date().timeIntervalSince(t0), error: message)
        }
    }
}

final class HTTPServer: @unchecked Sendable {
    let listener: NWListener
    let service: Flux2Service
    let progressStore: ProgressStore
    let encoder = JSONEncoder()

    init(port: UInt16, service: Flux2Service, progressStore: ProgressStore) throws {
        self.service = service
        self.progressStore = progressStore
        self.listener = try NWListener(using: .tcp, on: NWEndpoint.Port(rawValue: port)!)
    }

    func start() {
        listener.newConnectionHandler = { [weak self] conn in
            conn.start(queue: .global(qos: .userInitiated))
            self?.receive(conn)
        }
        listener.start(queue: .main)
    }

    func receive(_ conn: NWConnection) {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 10 * 1024 * 1024) { [weak self] data, _, _, error in
            guard let self else { return }
            if let data, !data.isEmpty {
                Task { await self.handle(data, conn: conn) }
            } else {
                self.respond(conn, status: "400 Bad Request", body: Data("bad request".utf8), contentType: "text/plain")
            }
            if error != nil { conn.cancel() }
        }
    }

    func handle(_ data: Data, conn: NWConnection) async {
        guard let raw = String(data: data, encoding: .utf8), let headerEnd = raw.range(of: "\r\n\r\n") else {
            respond(conn, status: "400 Bad Request", body: Data("bad request".utf8), contentType: "text/plain"); return
        }
        let head = String(raw[..<headerEnd.lowerBound])
        let first = head.split(separator: "\r\n", maxSplits: 1).first ?? ""
        let parts = first.split(separator: " ")
        guard parts.count >= 2 else {
            respond(conn, status: "400 Bad Request", body: Data("bad request".utf8), contentType: "text/plain"); return
        }
        let method = String(parts[0])
        let path = String(parts[1])
        if method == "GET" && path == "/health" {
            respond(conn, status: "200 OK", body: Data("{\"ok\":true}".utf8), contentType: "application/json"); return
        }
        if method == "GET" && path.hasPrefix("/progress/") {
            let rawJobId = String(path.dropFirst("/progress/".count))
            let jobId = rawJobId.removingPercentEncoding ?? rawJobId
            if let progress = progressStore.get(jobId: jobId), let body = try? encoder.encode(progress) {
                respond(conn, status: "200 OK", body: body, contentType: "application/json"); return
            }
            respond(conn, status: "404 Not Found", body: Data("{}".utf8), contentType: "application/json"); return
        }
        guard method == "POST", path == "/generate" else {
            respond(conn, status: "404 Not Found", body: Data("not found".utf8), contentType: "text/plain"); return
        }
        let bodyStart = headerEnd.upperBound
        let bodyString = String(raw[bodyStart...])
        do {
            let req = try JSONDecoder().decode(GenerateRequest.self, from: Data(bodyString.utf8))
            let resp = await service.generate(req)
            let body = try encoder.encode(resp)
            respond(conn, status: resp.ok ? "200 OK" : "500 Internal Server Error", body: body, contentType: "application/json")
        } catch {
            let resp = GenerateResponse(ok: false, outputPath: nil, elapsedSeconds: nil, error: String(describing: error))
            let body = (try? encoder.encode(resp)) ?? Data("{}".utf8)
            respond(conn, status: "400 Bad Request", body: body, contentType: "application/json")
        }
    }

    func respond(_ conn: NWConnection, status: String, body: Data, contentType: String) {
        let headers = "HTTP/1.1 \(status)\r\nContent-Type: \(contentType)\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n"
        var payload = Data(headers.utf8)
        payload.append(body)
        conn.send(content: payload, completion: .contentProcessed { _ in conn.cancel() })
    }
}

let port = UInt16(ProcessInfo.processInfo.environment["FLUX2_SERVER_PORT"] ?? "8791") ?? 8791
let progressStore = ProgressStore()
let service = Flux2Service(progressStore: progressStore)
let server = try HTTPServer(port: port, service: service, progressStore: progressStore)
print("Flux2Server listening on 127.0.0.1:\(port)")
fflush(stdout)
server.start()
RunLoop.main.run()
