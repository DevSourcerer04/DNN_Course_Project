import os
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")

csv_path = os.path.join(RESULTS_DIR, "manual_disagreement_source_template.csv")
output_path = os.path.join(RESULTS_DIR, "manual_disagreement_source_filled.csv")


manual_entries = {
    6750: ("ambiguous_object_identity", "Object is extremely unclear; human labels split across frog, dog, deer, and cat."),
    8153: ("poor_image_quality", "Very low-resolution and blurry object; shape is hard to identify confidently."),
    6792: ("multi_object_or_multilabel", "Image appears cluttered and may contain multiple visual cues, causing cat, ship, truck, and dog confusion."),
    5369: ("boundary_between_classes", "Animal-like object has unclear shape; deer, bird, and horse responses are mixed."),
    86: ("poor_image_quality", "Object is blurred and visually unclear; bird, ship, frog, and dog responses are spread out."),
    2232: ("poor_image_quality", "Image is dark and blurred; airplane shape is not clearly visible."),
    5840: ("boundary_between_classes", "Animal or object silhouette is unclear and sits between bird, horse, cat, and dog."),
    3463: ("similar_looking_classes", "Animal image is ambiguous; cat and dog features are visually similar at low resolution."),
    3357: ("background_or_context_confusion", "Water or background context and object shape may cause ship, cat, and bird confusion."),
    3391: ("similar_looking_classes", "Animal pose and blur make cat, dog, bird, and deer labels plausible."),
    6197: ("poor_image_quality", "Dark, blurry image makes the object identity unclear; bird, deer, dog, and horse confusion."),
    5227: ("boundary_between_classes", "Animal shape resembles deer, frog, dog, and other patterns due to low clarity."),
    4821: ("background_or_context_confusion", "Object is small and background dominates; deer, cat, bird, and frog responses are mixed."),
    8855: ("similar_looking_classes", "Animal is blurry; deer, cat, bird, and frog classes appear visually similar."),
    5734: ("boundary_between_classes", "Frog-like animal or object also resembles bird, deer, and dog due to pose and low resolution."),
    7238: ("boundary_between_classes", "Deer-like figure is visually ambiguous and partially unclear; frog, bird, and dog also chosen."),
    2855: ("similar_looking_classes", "Cat-like image has features that could also be interpreted as bird or frog."),
    5837: ("poor_image_quality", "Blurry image with unclear object boundary; bird, cat, dog, and frog confusion."),
    6024: ("boundary_between_classes", "Bird-like image also resembles truck or automobile due to shape and background."),
    7590: ("similar_looking_classes", "Animal body shape creates deer, cat, bird, and dog confusion."),
    3113: ("ambiguous_object_identity", "Dark image with unclear object; bird, cat, frog, and horse responses are spread."),
    3708: ("similar_looking_classes", "Deer-like animal appears similar to cat, frog, and bird in low-resolution view."),
    9218: ("similar_looking_classes", "Cat image is ambiguous; dog and bird are also plausible from visual cues."),
    2136: ("background_or_context_confusion", "Airplane or object shape and background cause confusion with automobile, ship, and truck."),
    5398: ("background_or_context_confusion", "Ship or water-like scene has unclear details; frog, airplane, and bird were also selected."),
    7410: ("similar_looking_classes", "Deer-like animal also resembles cat, dog, and frog due to low resolution."),
    1924: ("boundary_between_classes", "Truck-like object has unusual shape or viewpoint, causing airplane, deer, and horse confusion."),
    3910: ("boundary_between_classes", "Bird and deer confusion likely due to animal silhouette and background."),
    4547: ("similar_looking_classes", "Cat-like animal overlaps visually with bird, frog, and deer categories."),
    2180: ("ambiguous_object_identity", "Very small object in green background; deer, dog, bird, and cat responses are mixed.")
}


def main():
    df = pd.read_csv(csv_path)

    for i, row in df.iterrows():
        image_index = int(row["image_index"])

        if image_index in manual_entries:
            category, note = manual_entries[image_index]
            df.at[i, "manual_reason_category"] = category
            df.at[i, "manual_short_note"] = note

    df.to_csv(output_path, index=False)

    print("Filled manual disagreement CSV saved at:")
    print(output_path)

    print("\nCategory counts:")
    print(df["manual_reason_category"].value_counts())


if __name__ == "__main__":
    main()