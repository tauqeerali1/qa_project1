from django.db import models

class Document(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
    
@classmethod
def create_test_document(cls):
    return cls.objects.create(
        title='Test Dataset',
        content='Python is a programming language. It was created by Guido van Rossum.'
    )