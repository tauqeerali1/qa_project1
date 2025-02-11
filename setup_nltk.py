import nltk

def download_nltk_resources():
    print("Downloading NLTK resources...")
    try:
        # Download the punkt tokenizer data
        nltk.download('punkt')
        print("Successfully downloaded NLTK resources!")
    except Exception as e:
        print(f"Error downloading NLTK resources: {str(e)}")

if __name__ == "__main__":
    download_nltk_resources()