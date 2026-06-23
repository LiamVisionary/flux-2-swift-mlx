import Foundation
import Network
import MLX
import Flux2Core
import FluxTextEncoders
import ImageIO
import UniformTypeIdentifiers
import CoreGraphics

extension MLXArray: @unchecked Sendable {}

struct GenerateRequest: Codable {
    var prompt: String
    var imagePath: String
    var outputPath: String
    var width: Int?
    var height: Int?
    var steps: Int?
    var guidance: Float?
    var seed: UInt64?
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
    let pipeline: Flux2Pipeline
    var embeddingCache: [String: MLXArray] = [:]

    init() {
        let textQuant = MistralQuantization.mlx8bit
        let transformerQuant = TransformerQuantization.bf16
        let quant = Flux2QuantizationConfig(textEncoder: textQuant, transformer: transformerQuant)
        let token = ProcessInfo.processInfo.environment["HF_TOKEN"]
        self.pipeline = Flux2Pipeline(model: .klein9B, quantization: quant, vaeVariant: .standard, hfToken: token)
        self.pipeline.memoryProfile = .performance
        Flux2Debug.setNormalMode()
        FluxDebug.isEnabled = false
    }

    func generate(_ req: GenerateRequest) async -> GenerateResponse {
        let t0 = Date()
        do {
            guard let loadedRef = loadImage(from: req.imagePath) else {
                throw Flux2Error.imageProcessingFailed("Failed to load reference image: \(req.imagePath)")
            }
            let targetWidth = req.width ?? loadedRef.width
            let targetHeight = req.height ?? loadedRef.height
            guard let ref = cropResizeImage(loadedRef, width: targetWidth, height: targetHeight) else {
                throw Flux2Error.imageProcessingFailed("Failed to prepare reference image")
            }
            let embeddings: MLXArray
            if let cached = embeddingCache[req.prompt] {
                embeddings = cached
            } else {
                embeddings = try await pipeline.precomputeTextEmbeddings(prompt: req.prompt, upsamplePrompt: false)
                embeddingCache[req.prompt] = embeddings
            }
            let image = try await pipeline.generate(
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
                onProgress: nil,
                onCheckpoint: nil,
                onStep: nil
            )
            try saveImage(image, to: req.outputPath)
            return GenerateResponse(ok: true, outputPath: req.outputPath, elapsedSeconds: Date().timeIntervalSince(t0), error: nil)
        } catch {
            return GenerateResponse(ok: false, outputPath: nil, elapsedSeconds: Date().timeIntervalSince(t0), error: String(describing: error))
        }
    }
}

final class HTTPServer: @unchecked Sendable {
    let listener: NWListener
    let service: Flux2Service
    let encoder = JSONEncoder()

    init(port: UInt16, service: Flux2Service) throws {
        self.service = service
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
let service = Flux2Service()
let server = try HTTPServer(port: port, service: service)
print("Flux2Server listening on 127.0.0.1:\(port)")
fflush(stdout)
server.start()
RunLoop.main.run()
