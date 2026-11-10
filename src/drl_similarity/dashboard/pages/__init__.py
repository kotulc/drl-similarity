"""Dashboard pages. Each module exposes ``render(data)`` and never reads files directly."""

PAGE_ORDER: tuple[tuple[str, str], ...] = (
    ("similarity_matrix", "Similarity matrix"),
    ("scenario_explorer", "Scenario explorer"),
    ("training_overview", "Training and quality"),
    ("scenario_separation", "Scenario separation"),
    ("cross_checkpoint", "Cross-checkpoint geometry"),
    ("provenance", "Provenance and exports"),
)
