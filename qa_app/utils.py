from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QASystem:
    def __init__(self):
        try:
            logger.info("Initializing QA System...")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
            self.model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
            self.max_length = 512
            self.stride = 128
            self.max_chunks = 10
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.context_window = 100
            logger.info("QA System initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing QA System: {str(e)}")
            raise

    def preprocess_text(self, text):
        """Clean and preprocess the text."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Ensure proper sentence endings
        text = text.replace('..', '.').replace('?.', '?').replace('!.', '!')
        return text

    def split_into_chunks(self, context):
        """Split text into manageable chunks."""
        try:
            context = self.preprocess_text(context)
            # Split by sentences while maintaining context
            sentences = []
            current_sentence = ""
            
            for char in context:
                current_sentence += char
                if char in ['.', '?', '!'] and len(current_sentence.strip()) > 0:
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
            
            if current_sentence.strip():
                sentences.append(current_sentence.strip())

            chunks = []
            current_chunk = []
            current_length = 0

            for sentence in sentences:
                tokens = self.tokenizer.encode(sentence)
                if len(tokens) > self.max_length:
                    # Split long sentences
                    words = sentence.split()
                    temp_sentence = ""
                    for word in words:
                        if len(self.tokenizer.encode(temp_sentence + " " + word)) > self.max_length:
                            if temp_sentence:
                                chunks.append(temp_sentence)
                            temp_sentence = word
                        else:
                            temp_sentence += " " + word if temp_sentence else word
                    if temp_sentence:
                        chunks.append(temp_sentence)
                else:
                    if current_length + len(tokens) > self.max_length:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = [sentence]
                        current_length = len(tokens)
                    else:
                        current_chunk.append(sentence)
                        current_length += len(tokens)

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            logger.info(f"Split text into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error in split_into_chunks: {str(e)}")
            # Fallback to simple chunking
            return [context[i:i+self.max_length] for i in range(0, len(context), self.max_length)]

    def select_relevant_chunks(self, question, chunks):
        """Select the most relevant chunks for the question."""
        try:
            # Add the question to all texts for vectorization
            all_texts = chunks + [question]
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Get question vector (last in matrix)
            question_vector = tfidf_matrix[-1]
            
            # Calculate similarities
            similarities = cosine_similarity(question_vector, tfidf_matrix[:-1])[0]
            
            # Get top chunk indices
            top_indices = np.argsort(similarities)[-self.max_chunks:]
            
            selected_chunks = [chunks[i] for i in top_indices]
            logger.info(f"Selected {len(selected_chunks)} most relevant chunks")
            return selected_chunks

        except Exception as e:
            logger.error(f"Error in select_relevant_chunks: {str(e)}")
            return chunks[:self.max_chunks]

    def get_context_window(self, text, answer_start, answer_end):
        """Get expanded context around the answer."""
        words = text.split()
        text_length = len(words)
        
        # Find word positions
        current_pos = 0
        start_word_idx = 0
        end_word_idx = 0
        
        for i, word in enumerate(words):
            word_length = len(word) + 1  # +1 for space
            if current_pos <= answer_start < current_pos + word_length:
                start_word_idx = i
            if current_pos <= answer_end < current_pos + word_length:
                end_word_idx = i
                break
            current_pos += word_length
        
        # Expand context window
        context_start = max(0, start_word_idx - self.context_window)
        context_end = min(text_length, end_word_idx + self.context_window + 1)
        
        return ' '.join(words[context_start:context_end])

    def process_chunk(self, args):
        """Process a single chunk to find answers."""
        chunk, question, chunk_id = args
        try:
            inputs = self.tokenizer.encode_plus(
                question,
                chunk,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                return_offsets_mapping=True
            )

            offset_mapping = inputs.pop('offset_mapping')[0]

            with torch.no_grad():
                outputs = self.model(**inputs)
            
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
            
            # Get best answer span
            start_idx = torch.argmax(start_scores)
            end_idx = torch.argmax(end_scores)
            
            if start_idx <= end_idx:
                start_char = int(offset_mapping[start_idx][0])
                end_char = int(offset_mapping[end_idx][1])
                
                if start_char < end_char:
                    answer_span = chunk[start_char:end_char]
                    extended_answer = self.get_context_window(chunk, start_char, end_char)
                    
                    return {
                        'answer': extended_answer,
                        'score': (start_scores[0][start_idx] + end_scores[0][end_idx]).item(),
                        'chunk_id': chunk_id,
                        'original_span': answer_span
                    }
            return None

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
            return None

    def get_answer(self, question, context):
        """Get the best answer for a question from the context."""
        try:
            logger.info(f"Processing question: {question}")
            
            if not context or len(context) < 10:
                return "No context provided or context too short."

            # Split into chunks
            chunks = self.split_into_chunks(context)
            
            # Select relevant chunks
            relevant_chunks = self.select_relevant_chunks(question, chunks)
            
            best_answer = ""
            best_score = float('-inf')
            answers = []

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(self.process_chunk, (chunk, question, i))
                    for i, chunk in enumerate(relevant_chunks)
                ]

                for future in as_completed(futures):
                    result = future.result()
                    if result and result['answer'] and not result['answer'].isspace():
                        answers.append(result)
                        if result['score'] > best_score:
                            best_answer = result['answer']
                            best_score = result['score']
                            logger.info(f"Found better answer in chunk {result['chunk_id']}")

            if not best_answer:
                return "I couldn't find a relevant answer in the provided text."

            return best_answer

        except Exception as e:
            logger.error(f"Error in get_answer: {str(e)}")
            return f"An error occurred while processing your question: {str(e)}"