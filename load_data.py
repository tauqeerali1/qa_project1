import os
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'qa_project.settings')
django.setup()

from qa_app.models import Document

def load_dataset():
    try:
        with open('atharvaved_cleaned.txt', 'r', encoding='utf-8') as file:
            content = file.read()
            Document.objects.create(
                title='Dataset',
                content=content
            )
        print("Dataset loaded successfully!")
    except FileNotFoundError:
        print("Error: atharvaved_cleaned.txt not found. Please make sure it's in the correct location.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    load_dataset()