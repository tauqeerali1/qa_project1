from django.shortcuts import render
from django.views.generic import TemplateView
from .forms import QuestionForm
from .utils import QASystem
from .models import Document
import logging

logger = logging.getLogger(__name__)

class HomeView(TemplateView):
    template_name = 'qa_app/home.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = QuestionForm()
        return context

    def post(self, request, *args, **kwargs):
        form = QuestionForm(request.POST)
        context = self.get_context_data()
        
        if form.is_valid():
            question = form.cleaned_data['question']
            try:
                logger.info(f"Processing question: {question}")
                
                # Get the content from the latest document
                document = Document.objects.latest('uploaded_at')
                logger.info(f"Found document: {document.title}")
                
                if not hasattr(self, 'qa_system'):
                    logger.info("Initializing QA System")
                    self.qa_system = QASystem()
                
                answer = self.qa_system.get_answer(question, document.content)
                logger.info(f"Got answer: {answer}")
                
                context['answer'] = answer
                context['question'] = question
                
            except Document.DoesNotExist:
                error_msg = "No document found in the database."
                logger.error(error_msg)
                context['error'] = error_msg
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                logger.error(error_msg)
                context['error'] = error_msg
        
        return render(request, self.template_name, context)