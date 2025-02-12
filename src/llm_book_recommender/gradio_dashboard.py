import pandas as pd
import numpy as np

from dotenv import load_dotenv
import gradio as gr

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# Load environment variables from a .env file
load_dotenv()

books = pd.read_csv("src/llm_book_recommender/books_with_emotions.csv")

# Create a new column for high-resolution thumbnails
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

# Handle missing thumbnails by replacing them with a default image
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Load raw documents from a text file
raw_documents = TextLoader("src/llm_book_recommender/tagged_description.txt").load()

# Initialize a text splitter to split documents by newline characters
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)

# Split the raw documents into smaller chunks
documents = text_splitter.split_documents(raw_documents)

# Initialize HuggingFace embeddings for semantic search
huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a Chroma database from the documents using the embeddings
db_books = Chroma.from_documents(documents, embedding=huggingface_embeddings)

def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k=50,
    final_top_k=16,
) -> pd.DataFrame:
    """
    Retrieve book recommendations based on a semantic search query.

    Parameters:
    - query (str): The search query.
    - category (str, optional): The category to filter books by.
    - tone (str, optional): The emotional tone to sort books by.
    - initial_top_k (int, optional): The number of top results to initially retrieve.
    - final_top_k (int, optional): The number of top results to return after filtering.

    Returns:
    - pd.DataFrame: A DataFrame containing the recommended books.
    """
    # Perform a similarity search on the database
    recs = db_books.similarity_search(query, k=initial_top_k)

    # Extract book identifiers from the search results
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]

    # Filter the books DataFrame to include only the recommended books
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    # Filter by category if specified
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category][:final_top_k]
    else:
        book_recs = book_recs.head(final_top_k)

    # Sort the recommendations by the specified emotional tone
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

def recommend_books(
    query: str,
    category: str,
    tone: str,
):
    """
    Generate a list of book recommendations based on user input.

    Parameters:
    - query (str): The search query.
    - category (str): The category to filter books by.
    - tone (str): The emotional tone to sort books by.

    Returns:
    - list: A list of tuples containing book thumbnails and captions.
    """
    # Retrieve recommendations based on the query, category, and tone
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    # Iterate over the recommended books to format the output
    for _, row in recommendations.iterrows():
        description = row["description"]
        # Truncate the description to the first 30 words
        truncated_description = " ".join(description.split()[:30]) + "..."

        # Format the authors' names
        author_split = row["authors"].split(";")
        if len(author_split) == 2:
            authors_str = f"{author_split[0]} and {author_split[1]}"
        elif len(author_split) > 2:
            authors_str = f"{', '.join(author_split[:-1])}, and {author_split[-1]}"
        else:
            authors_str = row["authors"]

        # Create a caption for the book
        caption = f"{row['title']} by {authors_str} : {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

# Define categories and tones for the dropdown menus
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Create the Gradio interface
with gr.Blocks() as dashboard:
    gr.Markdown("# Book Recommender System")
    gr.Markdown("### Find your next favorite book based on your preferences")

    with gr.Row():
        with gr.Column(scale=2):
            user_query = gr.Textbox(label="Please enter a description of a book:",
                                    placeholder="e.g., A story about love and dragons")

            category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")

            tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotion tone:", value="All")
            
            submit_button = gr.Button("Find Recommendations")

        with gr.Column(scale=3):
            gr.Markdown("## Recommendations")
            output = gr.Gallery(label="Recommended Books", columns=4, rows=4, object_fit="contain")

    # Set up the button click event to trigger the recommendation function
    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

# Launch the Gradio dashboard if this script is run directly
if __name__ == "__main__":
    dashboard.launch(share=True)