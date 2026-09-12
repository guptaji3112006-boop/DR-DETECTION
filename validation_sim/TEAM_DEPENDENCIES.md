# Team Dependencies Status (Member 4 - Systems / Validation Lead)

## Member 1 (IQA / Triage)
- **Split-specific tuning manifest/hashes**: Received (Provenance CSV supplied. 5 tuning images [IDRiD_001.jpg - IDRiD_005.jpg] confirmed to have no exact-file hash matches in the Testing manifest).
- **Batch retuning clarification**: Received (Developer reports thresholds were NOT changed after the 413-image Training batch evaluation).
- **Full Training/Testing hash comparison**: Verified (One matching exact-file pair identified: Training IDRiD_118.jpg and Testing IDRiD_064.jpg share SHA-256 505330878f4d4186a71b8001570100b82b6b05cb431e81e2c259d8c584314708).
- **Actual-file confirmation**: Verified (The exact matching pair was confirmed via local rehashing; Member 1 also reported this verification).
- **Final IQA version**: Received (Commit c884e85 incorporated; missing-quadrant uniformity calculation fixed to return 0.0).
- **Agreed recapture policy**: Pending (Current simulation configures a maximum of 2 capture attempts, pending team agreement to resolve documentation mismatch).

## Member 2 (Classifier)
- **Checkpoint identity**: Received (diabetic_retinopathy_model.keras).
- **Training/calibration manifests**: Pending (Reported APTOS-trained. Classifier training/calibration independence from IDRiD remains unverified).
- **Preprocessing/class mapping**: Developer-reported (Original image, RGB, 224x224, [0, 1] values. Class mapping: 0, 1, 2, 3, 4 based on argmax).
- **Actual calibration artifacts**: Pending (Raw confidence is currently uncalibrated).
- **Lesion-aware model status**: Pending.

## Member 3 (Explainability)
- **Actual-checkpoint Grad-CAM**: Pending.
- **Lesion-mask provenance**: Pending.
- **Real ablation results**: Pending.

## Member 5 (UI/Dashboard)
- **Selected-run consumption**: Pending (Runs explicitly defined in `validation_sim/config/selected_runs.json`).
- **Rejection handling**: Pending (Must handle `SKIPPED_BY_GATE` abstentions with no displayed severity grade).
- **Correct confidence states**: Pending (Missing values/N/A must not be cast to 0. Probabilities must be parsed as JSON, not evaluated. Missing calibration must show as unavailable).
- **Integration confirmation**: Pending.
