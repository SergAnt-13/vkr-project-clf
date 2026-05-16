from hierarchical_semantic_pipeline import HierarchicalSemanticPipeline

pipeline = HierarchicalSemanticPipeline()
for query in ["битки с луком", "голубцы", "йогурт клубничный", "минтай ломтики"]:
    result = pipeline.classify_product(query)
    print(f"{query} -> {result}")