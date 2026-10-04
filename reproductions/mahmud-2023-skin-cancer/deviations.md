# Deviations and unresolved reproducibility details

This initial record comes from inspecting the imported notebooks, not a full audit
against the paper and upstream code. Pin upstream versions before confirming fidelity.

| Component | EfficientNetV2S reproduction | Xception extension | Comparison impact |
|---|---|---|---|
| Array construction | Reconstructs unpublished arrays from metadata order, RGB uint8, PIL bilinear 224×224 resize | Streams decoded/resized images with tf.data | Original preprocessing/order not yet verified |
| Split | Image-level stratified 8,111 / 902 / 1,002 split | StratifiedGroupKFold by lesion_id; saved test has 992 images | Different test sets; Xception is not a direct accuracy replication |
| Training | Full backbone, 50 epochs, Adam 0.001 | Frozen-head stage then partial fine-tuning, class weights, early stopping | Xception changes optimization and selection |
| Augmentation | ImageDataGenerator including shear | Keras augmentation layers, shear omitted | Augmentation differs |
| Added analysis | Faster Score-CAM | Entropy, referral, extra metrics, SmoothGrad | Uncertainty/referral is an extension |
| Environment | Assumed seed 42; runtime dependencies installed without a complete lock | Seed 42; runtime dependency installation | Exact original environment not established |
| Saved execution | Score-CAM NameError; final export unexecuted | Evaluation and export outputs present | Imported outputs require fresh validation |

Preserve the image-level split when reproducing the released experiment, but report
its lesion-overlap risk. Keep the lesion-safe evaluation explicitly separate.
