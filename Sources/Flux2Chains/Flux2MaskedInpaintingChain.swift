// Flux2MaskedInpaintingChain.swift — RePaint-style masked inpainting
// Copyright 2025 Vincent Gourbin
//
// FLUX.2 has no dedicated Fill checkpoint (as of 2026-05). Diffusers'
// `FluxFillPipeline` channel-cat approach requires fine-tuned input channels,
// which FLUX.2 base/distilled doesn't have. The classical RePaint trick gets
// us most of the way for free: at every denoising step the region OUTSIDE the
// user mask is overwritten with the original-image latent re-noised to the
// current sigma. Only the inside-mask region accumulates new content.
//
// Implemented as a `Flux2Chain` using the `onStep:` hook on the standard T2I
// generation path — so progress callbacks, profiling, memory caps, and
// scheduler overrides all behave exactly like a normal generate() call.

import Foundation
import Flux2Core
@preconcurrency import MLX
@preconcurrency import MLXRandom
import CoreGraphics

/// Masked inpainting chain (RePaint-style per-step latent blending).
public struct Flux2MaskedInpaintingChain: Flux2Chain {
    /// Underlying pipeline. The chain does not own it — host apps can reuse
    /// the same pipeline across many chain runs without paying the model
    /// load cost each time.
    public let pipeline: Flux2Pipeline

    /// Inputs.
    public let prompt: String
    public let image: CGImage
    /// Grayscale mask, same dimensions as `image` (resized otherwise).
    /// White = inpaint, black = keep. Soft values in [0, 1] are honored.
    public let mask: CGImage
    /// Optional reference image(s) for the transformer to attend to in
    /// addition to the prompt. When provided, the chain switches generation
    /// from `.textToImage` to `.imageToImage(referenceImages)` while still
    /// applying the RePaint blend. Use this for *outpainting* where you want
    /// the painted strips to genuinely continue the visual content of the
    /// keep region — pass the extended canvas itself as the reference so the
    /// model sees what it should match. For pure intra-image inpainting this
    /// is unnecessary and adds compute.
    public let referenceImages: [CGImage]?
    public let steps: Int
    public let guidance: Float
    public let seed: UInt64?
    /// Optional progress callback forwarded to `generateWithResult`.
    public let onProgress: Flux2ProgressCallback?

    /// Maximum total pixel count for the working resolution. Larger inputs are
    /// scaled down (preserving aspect) before being clamped to a multiple of
    /// 32. Default 1024² matches the existing I2I conventions.
    public let maxPixels: Int

    public init(
        pipeline: Flux2Pipeline,
        prompt: String,
        image: CGImage,
        mask: CGImage,
        referenceImages: [CGImage]? = nil,
        steps: Int = 4,
        guidance: Float = 1.0,
        seed: UInt64? = nil,
        maxPixels: Int = 1024 * 1024,
        onProgress: Flux2ProgressCallback? = nil
    ) {
        self.pipeline = pipeline
        self.prompt = prompt
        self.image = image
        self.mask = mask
        self.referenceImages = referenceImages
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.maxPixels = maxPixels
        self.onProgress = onProgress
    }

    public func run() async throws -> Flux2GenerationResult {
        try await pipeline.loadModels()

        let (targetH, targetW) = Flux2Pipeline.resolveChainDimensions(
            width: image.width,
            height: image.height,
            maxPixels: maxPixels
        )

        // Encode the source image *once*, before the denoising loop starts.
        // The VAE stays resident so the post-denoising decode reuses it.
        let imageLatents = try await pipeline.encodeImageToPackedSequence(
            image,
            targetHeight: targetH,
            targetWidth: targetW
        )

        let maskLatents = Flux2Pipeline.packMaskForLatentBlending(
            mask,
            targetHeight: targetH,
            targetWidth: targetW
        )

        // RePaint blend: outside-mask region is forced back to (image latent
        // re-noised to sigmaNext). On the final step sigmaNext == 0 ⇒ the
        // original clean latent is restored (no hallucination outside mask).
        //
        // NOTE: the I2I path emits transformer noise predictions for the
        // *concatenated* (output + reference) sequence and slices them back
        // to the output portion before calling the hook, so this blend acts
        // exclusively on the output latents in both modes.
        let imageLatentsCaptured = imageLatents
        let maskLatentsCaptured = maskLatents
        let onStep: Flux2StepHook = { ctx, latents in
            let freshNoise = MLXRandom.normal(latents.shape)
            let sigmaNext = MLXArray(ctx.sigmaNext)
            let originalNoised = (1 - sigmaNext) * imageLatentsCaptured + sigmaNext * freshNoise
            return (1 - maskLatentsCaptured) * originalNoised + maskLatentsCaptured * latents
        }

        let mode: Flux2GenerationMode
        if let refs = referenceImages, !refs.isEmpty {
            mode = .imageToImage(images: refs)
        } else {
            mode = .textToImage
        }

        return try await pipeline.generateWithResult(
            mode: mode,
            prompt: prompt,
            interpretImagePaths: nil,
            height: targetH,
            width: targetW,
            steps: steps,
            guidance: guidance,
            seed: seed,
            upsamplePrompt: false,
            precomputedEmbeddings: nil,
            checkpointInterval: nil,
            onProgress: onProgress,
            onCheckpoint: nil,
            onStep: onStep
        )
    }
}
