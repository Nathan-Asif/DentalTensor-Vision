from dentaltensor import DentalTensorVision


def main():
    model = DentalTensorVision(device="cpu")

    image = "Test Images/Yellowish-Teeth.jpg"

    result = model.predict(
        image,
        threshold=0.01,
    )

    print("\n=== DentalTensor Vision v1.0 ===")
    print(f"Model: {result.model}")
    print(f"Version: {result.version}")
    print(f"Inference: {result.inference_ms} ms")

    print("\n=== Findings ===")

    if not result.findings:
        print("No findings.")
    else:
        for finding in result.findings:
            print(
                f"{finding.class_name}: "
                f"{finding.confidence:.4f} "
                f"(threshold={finding.threshold_used})"
            )

    print("\n=== JSON ===")
    print(result.to_json(indent=2))


if __name__ == "__main__":
    main()