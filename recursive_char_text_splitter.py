from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

tesla_text = """Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance

The Model Y became the best-selling vehicle globally, with 350,000 units sold.

Production Challenges

Supply chain issues caused a 12% increase in production costs.

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever making it impossible to split properly."""



print("\n" + "=" * 60)
print("1. CHARACTER TEXT SPLITTER SOLUTION")
print("=" * 60)

#example one noraml splitter

character_splitter = CharacterTextSplitter(
    separator = " ",
    chunk_size = 100,
    chunk_overlap = 0
)

character_chunks = character_splitter.split_text(tesla_text)
for i , chunk in enumerate(character_chunks , 1):
    print(f"Chunk {i} : ({len(chunk)} chars)")
    print(f"{chunk}")
    print()


print("\n" + "=" * 60)
print("2. RECURSIVE CHARACTER TEXT SPLITTER SOLUTION")
print("=" * 60)


#example two recursive splitter

recursive_splitter = RecursiveCharacterTextSplitter(
    separators = ["\n\n" , "\n" , ". " , "," , " "],
    chunk_size = 100,
    chunk_overlap = 0
)

recursive_chunks = recursive_splitter.split_text(tesla_text)
for i , chunk in enumerate(recursive_chunks , 1):
    print(f"Chunk {i} : ({len(chunk)} chars)")
    print(f"{chunk}")
    print()