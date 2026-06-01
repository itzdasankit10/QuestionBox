import asyncio
import json
import re
import os
import traceback
from io import BytesIO
from typing import Dict, List, Union, Callable, Optional
from datetime import datetime

# --- IMPORTS FOR API, FILE UPLOAD & OCR ---
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError, field_validator
from PIL import Image
import pytesseract
import pypdf

# --- IMPORTS FOR METRICS & TEXT PROCESSING ---
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

import ollama
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from openai import AsyncOpenAI
from dotenv import load_dotenv
from difflib import SequenceMatcher

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Configuring Hugging Face models...")

# --- INITIALIZE NLTK & METRICS ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK wordnet...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Initialize ROUGE scorer globally
rouge_evaluator = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# --- GLOBAL MODEL VARIABLES ---
ollama_client = None
openai_client = None
qa_pipeline = None
sentence_transformer = None
summarizer_pipeline = None
t5_qg_model = None
gpt2_pipeline = None

# --- INITIALIZE CLIENTS ---

# 1. OLLAMA
try:
    ollama_client = ollama.AsyncClient()
    print("Ollama client initialized.")
except Exception as ex:
    print(f"Error initializing Ollama client: {ex}")
    ollama_client = None

# 2. OPENAI
try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found in .env file. OpenAI models will not work.")
        openai_client = None
    else:
        openai_client = AsyncOpenAI(api_key=openai_api_key)
        print("OpenAI client initialized.")
except Exception as ex:
    print(f"Error initializing OpenAI client: {ex}")
    openai_client = None

# Check GPU Availability
use_cuda = torch.cuda.is_available()
hf_pipeline_device = 0 if use_cuda else -1
st_device_str = "cuda" if use_cuda else "cpu"
print(f"torch.cuda.is_available() = {use_cuda}. Pipeline device={hf_pipeline_device}, ST device={st_device_str}")

# --- LOAD LOCAL HUGGING FACE MODELS ---
try:
    # 3. QA MODEL
    preferred_qa = "deepset/bert-base-cased-squad2"
    fallback_qa = "distilbert-base-cased-distilled-squad"
    try:
        print(f"Loading preferred QA model: {preferred_qa}")
        qa_pipeline = pipeline("question-answering", model=preferred_qa, device=hf_pipeline_device)
        print(f"Loaded QA model: {preferred_qa}")
    except Exception as ex:
        print(f"Could not load preferred QA ({preferred_qa}): {ex}\nFalling back to {fallback_qa}")
        try:
            qa_pipeline = pipeline("question-answering", model=fallback_qa, device=hf_pipeline_device)
            print(f"Loaded fallback QA model: {fallback_qa}")
        except Exception as ex2:
            print(f"Failed to load fallback QA model ({fallback_qa}): {ex2}")
            qa_pipeline = None

    # 4. SENTENCE TRANSFORMER (Metrics)
    try:
        print("Loading SentenceTransformer (all-MiniLM-L6-v2)...")
        sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2", device=st_device_str)
        print("SentenceTransformer loaded.")
    except Exception as ex:
        print(f"Failed to load SentenceTransformer: {ex}")
        traceback.print_exc()
        sentence_transformer = None

    # 5. SUMMARIZER
    preferred_sum = "facebook/bart-large-cnn"
    fallback_sum = "sshleifer/distilbart-cnn-12-6"
    try:
        print(f"Loading summarizer: {preferred_sum}")
        summarizer_pipeline = pipeline("summarization", model=preferred_sum, device=hf_pipeline_device)
        print(f"Loaded summarizer: {preferred_sum}")
    except Exception as ex:
        print(f"Could not load preferred summarizer ({preferred_sum}): {ex}\nFalling back to {fallback_sum}")
        try:
            summarizer_pipeline = pipeline("summarization", model=fallback_sum, device=hf_pipeline_device)
            print(f"Loaded fallback summarizer: {fallback_sum}")
        except Exception as ex2:
            print(f"Failed to load fallback summarizer ({fallback_sum}): {ex2}")
            summarizer_pipeline = None

    # 6. T5 GENERATOR
    # preferred_t5 = "voidful/t5-base-qg-hl"
    # fallback_t5 = "valhalla/t5-small-qg-hl"

    preferred_t5 = r"Y:\ai_question\New folder\flan-t5-question-generator"
    fallback_t5 = None
    try:
        print(f"Attempting T5 QG model: {preferred_t5}")
        t5_tokenizer = T5Tokenizer.from_pretrained(preferred_t5)
        t5_model = T5ForConditionalGeneration.from_pretrained(preferred_t5)
        t5_qg_model = {"tokenizer": t5_tokenizer, "model": t5_model, "device": st_device_str}
        if use_cuda:
            t5_model.to("cuda")
        else:
            t5_model.to("cpu")
        print(f"Loaded T5 QG: {preferred_t5}")
    except Exception as ex:
        print(f"Preferred T5 failed: {ex}\nTrying fallback T5: {fallback_t5}")
        try:
            t5_tokenizer = T5Tokenizer.from_pretrained(fallback_t5)
            t5_model = T5ForConditionalGeneration.from_pretrained(fallback_t5)
            t5_qg_model = {"tokenizer": t5_tokenizer, "model": t5_model, "device": st_device_str}
            if use_cuda:
                t5_model.to("cuda")
            else:
                t5_model.to("cpu")
            print(f"Loaded fallback T5 QG: {fallback_t5}")
        except Exception as ex2:
            print(f"Failed to load any T5 QG model: {ex2}")
            traceback.print_exc()
            t5_qg_model = None

    # 7. GPT-2 GENERATOR
    try:
        print("Loading GPT-2 generator (gpt2)...")
        gpt2_pipeline = pipeline("text-generation", model="gpt2", device=hf_pipeline_device)
        print("GPT-2 generator loaded.")
    except Exception as ex:
        print(f"Failed to load GPT-2 generator: {ex}")
        traceback.print_exc()
        gpt2_pipeline = None

    print("Model loading complete.")
except Exception:
    print("Unexpected error during model initialization:")
    traceback.print_exc()

# --- FILE PATH CONFIGURATION ---
BOOKS_BASE_DIR = "BOOKS"
QUESTIONS_BASE_DIR = "QUESTIONS"
GENERATED_PAPERS_DIR = "GENERATED_PAPERS"  # NEW: Directory to save generated papers

if not os.path.exists(BOOKS_BASE_DIR):
    try:
        os.makedirs(BOOKS_BASE_DIR)
        print(f"Created base directory: {BOOKS_BASE_DIR}")
    except OSError as ex:
        print(f"Error creating directory {BOOKS_BASE_DIR}: {ex}")
        BOOKS_BASE_DIR = "."

@app.get("/generated-papers")
async def get_generated_papers():
    papers = []

    try:
        if not os.path.exists(GENERATED_PAPERS_DIR):
            return []

        for filename in os.listdir(GENERATED_PAPERS_DIR):

            if filename.endswith(".json"):

                filepath = os.path.join(GENERATED_PAPERS_DIR, filename)

                try:
                    with open(filepath, "r", encoding="utf-8") as f:

                        data = json.load(f)

                        papers.append({
                            "id": filename,
                            "filename": filename,
                            "timestamp": data.get("metadata", {}).get("timestamp", ""),
                            "classNum": data.get("metadata", {}).get("class_num", ""),
                            "subject": data.get("metadata", {}).get("subject", ""),
                            "model": data.get("metadata", {}).get("model_name", ""),
                            "questions": data.get("questions", {}),
                            "context": data.get("context", "")
                        })

                except Exception as e:
                    print(f"Error reading {filename}: {e}")

        return papers

    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

if not os.path.exists(QUESTIONS_BASE_DIR):
    try:
        os.makedirs(QUESTIONS_BASE_DIR)
        print(f"Created questions directory: {QUESTIONS_BASE_DIR}")
    except OSError as ex:
        print(f"Error creating directory {QUESTIONS_BASE_DIR}: {ex}")
        QUESTIONS_BASE_DIR = "."

# NEW: Create directory for generated papers
if not os.path.exists(GENERATED_PAPERS_DIR):
    try:
        os.makedirs(GENERATED_PAPERS_DIR)
        print(f"Created generated papers directory: {GENERATED_PAPERS_DIR}")
    except OSError as ex:
        print(f"Error creating directory {GENERATED_PAPERS_DIR}: {ex}")
        GENERATED_PAPERS_DIR = "."

SUBJECT_ABBREVIATIONS = {
    "Geography": "geo",
    "History": "his",
    "Economics": "eco",
    "English": "eng",
    "Biology": "bio",
    "Political Science": "pol",
    "Science": "science"
}

# --- DATA MODELS ---
class QuestionCounts(BaseModel):
    mcqs: int
    fill_in_the_blanks: int
    subjective: int

    @field_validator('mcqs', 'fill_in_the_blanks', 'subjective')
    def must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError('Question count must be non-negative')
        return v

class QuestionItem(BaseModel):
    question: str
    answer: Union[str, None] = None

class QuestionsPayload(BaseModel):
    questions: Dict[str, List[QuestionItem]]

class PdfRequest(BaseModel):
    answered_questions: Dict[str, List[QuestionItem]]
    model_name: Optional[str] = "Unknown Model"

class QuestionEvaluationRequest(BaseModel):
    question: str
    class_num: Optional[str] = None
    subject: Optional[str] = None
    custom_context: Optional[str] = None

class HumanEvaluationLog(BaseModel):
    question: str
    q_type: str
    evaluation: str

class QuestionRating(BaseModel):
    difficulty: str = ""
    remarks: str = ""

class InvigilatorRatingData(BaseModel):
    ratings: Dict[str, Dict[str, QuestionRating]] = {}
    overall_remarks: str = ""

class DifficultyRatings(BaseModel):
    class_num: Optional[str] = None
    subject: Optional[str] = None
    invigilator_data: Dict[str, InvigilatorRatingData]

# --- UTILITY FUNCTIONS ---

def save_generated_paper(
    questions: Dict[str, List[Dict]],
    context: str,
    model_name: str,
    class_num: Optional[str] = None,
    subject: Optional[str] = None,
    source_filename: Optional[str] = None
) -> str:
    """
    Save generated question paper to disk with metadata.
    Returns the saved file path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine base filename
    if source_filename:
        # Extract original filename without extension
        base_name = os.path.splitext(source_filename)[0]
        # Clean filename (remove special characters)
        base_name = re.sub(r'[^\w\-_]', '_', base_name)
        filename = f"{base_name}_{model_name}_{timestamp}.json"
    elif class_num and subject:
        filename = f"Class{class_num}_{subject}_{model_name}_{timestamp}.json"
    else:
        filename = f"custom_{model_name}_{timestamp}.json"
    
    filepath = os.path.join(GENERATED_PAPERS_DIR, filename)
    
    # Prepare data to save
    data = {
        "metadata": {
            "timestamp": timestamp,
            "model_name": model_name,
            "class_num": class_num,
            "subject": subject,
            "source_filename": source_filename,
            "generation_date": datetime.now().isoformat()
        },
        "questions": questions,
        "context": context
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved generated paper: {filename}")
        return filepath
    except Exception as ex:
        print(f"❌ Error saving generated paper: {ex}")
        traceback.print_exc()
        return ""

def get_context_from_selection(class_num: str, subject: str) -> str:
    """Load context from textbook files based on class and subject"""
    subject_abbr = SUBJECT_ABBREVIATIONS.get(subject)
    if not subject_abbr:
        raise HTTPException(status_code=404, detail=f"Subject '{subject}' is not configured.")
    
    class_dir = os.path.join(BOOKS_BASE_DIR, f"CLASS {class_num}")
    if not os.path.isdir(class_dir):
        raise HTTPException(status_code=404, detail=f"Directory for Class {class_num} not found.")
    
    file_prefix = f"class_{class_num}_{subject_abbr}"
    found_file_path = None
    
    try:
        for filename in os.listdir(class_dir):
            if filename.lower().startswith(file_prefix.lower()) and filename.lower().endswith(".txt"):
                found_file_path = os.path.join(class_dir, filename)
                break
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Base directory '{BOOKS_BASE_DIR}' not found.")
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Error accessing class directory: {ex}")
    
    if not found_file_path:
        raise HTTPException(
            status_code=404,
            detail=f"Textbook file for Class {class_num}, Subject {subject} not found."
        )
    
    try:
        with open(found_file_path, 'r', encoding='utf-8') as fh:
            content = fh.read()
            return content
    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading file '{os.path.basename(found_file_path)}': {ex}"
        )

def parse_reference_questions_from_text(a: str) -> List[str]:
    b = []
    a = a.strip()
    
    c = re.split(r'(?:^|\n|\s|[-_])\s*(?:Q\s*\.?\s*)?(\d+)[\.\)]\s+', a)
    
    if len(c) > 2:
        for d in range(1, len(c), 2):
            if d + 1 < len(c):
                e = c[d + 1].strip()
                f = re.split(r'\n\s*(?:Answer|Ans|Solution|A)[:.\s]', e, flags=re.IGNORECASE)[0]
                f = f.strip()
                f = re.sub(r'^(?:Q\s*\.?\s*)?\d+[\.\):\s]+', '', f)
                
                if f and len(f) > 10:
                    b.append(f)
    else:
        c = re.split(r'\n\s*\n+', a)
        for g in c:
            h = g.strip()
            if any(i in h.lower() for i in ['chapter', 'exercise', 'textbook', 'questions', 'section']):
                if len(h) < 50:
                    continue
            
            h = re.sub(r'^(?:Q\s*\.?\s*)?\d+[\.\):\s]+', '', h)
            h = re.split(r'\n\s*(?:Answer|Ans|Solution|A)[:.\s]', h, flags=re.IGNORECASE)[0]
            h = h.strip()
            
            if h and len(h) > 10:
                b.append(h)
                
    return b

def load_reference_questions(class_num: str, subject: str = None) -> List[str]:
    """Load reference questions from QUESTIONS directory"""
    reference_questions = []
    
    try:
        class_dir = os.path.join(QUESTIONS_BASE_DIR, f"CLASS {class_num}")
        
        if not os.path.isdir(class_dir):
            print(f"Warning: Questions directory for Class {class_num} not found at {class_dir}")
            return []
        
        files_to_process = []
        
        if subject:
            subject_dir = os.path.join(class_dir, subject)
            
            if os.path.isdir(subject_dir):
                print(f"Loading questions from hierarchical structure: {subject_dir}")
                for root, dirs, files in os.walk(subject_dir):
                    for filename in files:
                        if filename.lower().endswith(".txt"):
                            files_to_process.append(os.path.join(root, filename))
            else:
                print(f"Trying flat structure for {subject}...")
                subject_abbr = SUBJECT_ABBREVIATIONS.get(subject)
                if subject_abbr:
                    file_pattern = f"class_{class_num}_{subject_abbr}"
                    for filename in os.listdir(class_dir):
                        if filename.lower().startswith(file_pattern.lower()) and filename.lower().endswith(".txt"):
                            files_to_process.append(os.path.join(class_dir, filename))
        else:
            print(f"Loading all questions for Class {class_num}...")
            for root, dirs, files in os.walk(class_dir):
                for filename in files:
                    if filename.lower().endswith(".txt"):
                        files_to_process.append(os.path.join(root, filename))
        
        for file_path in files_to_process:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    questions = parse_reference_questions_from_text(content)
                    reference_questions.extend(questions)
                    print(f"✓ Loaded {len(questions)} questions from {os.path.basename(file_path)}")
            except Exception as e:
                print(f"✗ Error reading {os.path.basename(file_path)}: {e}")
    
    except Exception as ex:
        print(f"Error loading reference questions: {ex}")
        traceback.print_exc()
    
    if reference_questions:
        print(f"📚 Total reference questions loaded: {len(reference_questions)}")
    else:
        print(f"⚠️ No reference questions found for Class {class_num}, Subject: {subject}")
    
    return reference_questions

def clean_question_for_comparison(question: str) -> str:
    """Clean a question for fair comparison"""
    clean_q = question.strip()
    
    clean_q = re.sub(r'^\d+[\.\):\s]+', '', clean_q)
    clean_q = re.sub(r'^(?:Q\s*\.?\s*)?\d+[\.\):\s]+', '', clean_q)
    
    if '\nA)' in clean_q or '\nA.' in clean_q:
        clean_q = re.split(r'\n[A-D][\.\)]\s', clean_q)[0]
    
    if '**Options:**' in clean_q:
        clean_q = clean_q.split('**Options:**')[0]
    
    prefixes_to_remove = [
        r'^Here.*?text:\s*',
        r'^Here is a (?:new )?question:\s*',
        r'^\*\*Question:\*\*\s*',
        r'^Question:\s*'
    ]
    for prefix in prefixes_to_remove:
        clean_q = re.sub(prefix, '', clean_q, flags=re.IGNORECASE)
    
    clean_q = re.sub(r'\*\*', '', clean_q)
    clean_q = re.sub(r'__', '', clean_q)
    clean_q = re.sub(r'\s+', ' ', clean_q)
    
    return clean_q.strip()

def find_most_similar_reference_question(
    generated_question: str,
    reference_questions: List[str],
    similarity_threshold: float = 0.25
) -> Optional[Dict[str, Union[str, float]]]:
    """Find the most similar reference question from the textbook"""
    if not reference_questions:
        return None
    
    clean_gen_q = clean_question_for_comparison(generated_question)
    
    best_match = None
    best_score = 0.0
    best_method = "none"
    
    if sentence_transformer:
        try:
            gen_embedding = sentence_transformer.encode(clean_gen_q, convert_to_tensor=True)
            ref_embeddings = sentence_transformer.encode(reference_questions, convert_to_tensor=True)
            
            similarities = util.pytorch_cos_sim(gen_embedding, ref_embeddings)[0]
            max_sim_idx = torch.argmax(similarities).item()
            max_sim_score = similarities[max_sim_idx].item()
            
            if max_sim_score > best_score:
                best_score = max_sim_score
                best_match = reference_questions[max_sim_idx]
                best_method = "BERT"
            
            print(f"🔍 BERT similarity: {max_sim_score:.3f} - {reference_questions[max_sim_idx][:60]}...")
        except Exception as ex:
            print(f"Error in semantic similarity matching: {ex}")
    
    for ref_q in reference_questions:
        clean_ref_q = clean_question_for_comparison(ref_q)
        score = SequenceMatcher(None, clean_gen_q.lower(), clean_ref_q.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = ref_q
            best_method = "String Match"
    
    if best_score >= similarity_threshold:
        print(f"✅ Best match ({best_method}): {best_score:.3f}")
        return {
            'question': best_match,
            'similarity_score': best_score,
            'match_method': best_method
        }
    
    print(f"❌ No good match found. Best score: {best_score:.3f} (threshold: {similarity_threshold})")
    return None

def extract_text_from_stream(file_obj: BytesIO, filename: str) -> str:
    """Extract text from uploaded files"""
    content = ""
    try:
        lower_name = filename.lower()
        if lower_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
            image = Image.open(file_obj)
            content = pytesseract.image_to_string(image)
        elif lower_name.endswith('.pdf'):
            pdf_reader = pypdf.PdfReader(file_obj)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    content += text + "\n"
        elif lower_name.endswith('.txt'):
            content = file_obj.read().decode('utf-8')
        else:
            return ""
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        traceback.print_exc()
        return ""
    return content

def run_t5_qg(context: str, model_dict: dict):
    """Run T5 question generation"""
    tokenizer = model_dict['tokenizer']
    model = model_dict['model']
    device = model_dict['device']
    
    sentences = re.split(r'[.!?] ', context)
    target_sentences = [s for s in sentences if len(s) > 50]
    
    if not target_sentences:
        target_sentences = sentences
    if not target_sentences:
        return "Could not find a sentence to use for question generation."
    
    target_sentence = target_sentences[len(target_sentences) // 2]
    highlighted_context = context.replace(target_sentence, f"<hl> {target_sentence} <hl>")
    input_text = f"generate question: {highlighted_context}"
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(**inputs, max_length=64)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return question.replace("question: ", "").strip()

def parse_questions_from_block(content_block: str) -> List[Dict[str, Union[str, None]]]:
    """Parse questions from LLM output text block"""
    q_list = []
    if not content_block:
        return q_list
    
    content_block = re.sub(
        r'(?:^|\n)\s*(?:\*\*)?Part \d.*?(?:\n|$)',
        '',
        content_block,
        flags=re.IGNORECASE
    )
    
    q_blocks = re.split(r'\n\s*(?=\d+\.)', content_block.strip())
    
    for block in q_blocks:
        if not block:
            continue
        if re.match(r'^\s*(?:\*\*)?Part', block, re.IGNORECASE):
            continue
        
        parts = re.split(r'\n\s*Answer:.*', block, flags=re.IGNORECASE | re.DOTALL)
        question_text = parts[0]
        
        clean_q = re.sub(r'^\d+\.\s*', '', question_text.strip()).strip()
        clean_q = re.sub(r'^\s*Here.*?:\s*\n?', '', clean_q, flags=re.IGNORECASE | re.DOTALL).strip()
        
        if "**Options:**" in clean_q:
            clean_q = clean_q.split("**Options:**")[0]
        
        if "Part " in clean_q and len(clean_q) < 20:
            continue
        
        if clean_q:
            q_list.append({"question": clean_q, "answer": None})
    
    return q_list

def sanitize_chunk_output(raw: str) -> str:
    """Clean up raw LLM output"""
    if not raw:
        return ""
    
    t = raw
    
    m = re.search(r'response\s*=\s*"(.*?)"(?:\s|$)', t, flags=re.DOTALL)
    if m:
        t = m.group(1)
    
    t = re.sub(r'(?i)observe\s+the\s+(?:set-up|figure|diagram|image).*?(?:\.|\?|\n)', '', t)
    t = re.sub(r'(?i)(look\s+at|refer\s+to|observe)\s+the\s+(?:figure|image|diagram|graph|table).*?(?:\.|\?|\n)', '', t)
    t = re.sub(r'(?i)(?:as\s+shown\s+)?in\s+(?:fig|figure)\.?\s*\d+(?:\.\d+)?', '', t)
    
    t = re.sub(
        r'(?i)(?:Snehal|Paheli|Boojho|Rohan|The farmer|The student)\s+(?:observed|noted|saw|wanted)',
        'It was observed',
        t
    )
    
    t = re.sub(
        r'\b(?:model|created_at|done|done_reason|total_duration|load_duration|prompt_eval_count|prompt_eval_duration|eval_count|eval_duration|thinking|context|response)\s*=\s*(?:\[[^\]]*\]|"[^"]*"|\'[^\']*\'|[^\s,;]+)',
        ' ',
        t
    )
    t = re.sub(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', ' ', t)
    t = re.sub(r'(?i)prompt[:=]\s*[^\\n]+', ' ', t)
    
    m2 = re.search(r'(?:^|\n)((?:\s*\d+\.\s.*(?:\n|$))+)', t)
    if m2:
        t2 = m2.group(1)
        lines = t2.strip().splitlines()
        filtered = [ln for ln in lines if re.match(r'^\s*\d+\.', ln)]
        if filtered:
            return "\n".join(filtered).strip()
    
    m3 = re.search(
        r'(Here are .*?questions:|Here are the .*?MCQs based on the provided context:)(.*)',
        t,
        flags=re.IGNORECASE|re.DOTALL
    )
    if m3:
        body = m3.group(2)
        m4 = re.search(r'((?:\s*\d+\.\s.*(?:\n|$))+)', body)
        if m4:
            return m4.group(1).strip()
        return re.sub(r'\s+', ' ', body).strip()
    
    t = re.sub(r'\b\w+:\s*\S+', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    t = re.sub(r'[ \t]{2,}', ' ', t)
    
    return t.strip()

def clean_ocr_context_for_generation(text: str) -> str:
    """Remove figure references and narrative names from context"""
    lines = text.splitlines()
    cleaned_lines = []
    
    forbidden_markers = [
        "fig.", "figure", "observe the", "activity", "diagram",
        "see fig", "look at the"
    ]
    
    for line in lines:
        lower_line = line.lower()
        if any(marker in lower_line for marker in forbidden_markers):
            continue
        if re.match(r'^\s*(Paheli|Boojho|Snehal|Rohan)\s+(says|thought|asked|wants)', line, re.IGNORECASE):
            continue
        if len(line.strip()) < 5:
            continue
        cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)

def chunk_text_for_gpt2(text: str, approx_chunk_chars: int = 1800) -> List[str]:
    """Chunk text into smaller pieces for GPT-2 processing"""
    text = text.strip()
    if not text:
        return []
    
    sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    chunks = []
    current = []
    curr_len = 0
    
    for s in sentences:
        slen = len(s) + 1
        if curr_len + slen > approx_chunk_chars and current:
            chunks.append(" ".join(current).strip())
            current = [s]
            curr_len = slen
        else:
            current.append(s)
            curr_len += slen
    
    if current:
        chunks.append(" ".join(current).strip())
    
    return chunks

async def _run_llm_call(model_name: str, prompt: str, max_tokens: int = 1024) -> str:
    """Unified LLM caller"""
    t = ""
    
    if model_name.startswith("gpt-"):
        if not openai_client:
            raise HTTPException(status_code=503, detail="OpenAI client is not available.")
        
        response = await openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful quiz-generation assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )
        t = response.choices[0].message.content.strip()
    
    elif model_name == "t5":
        if not t5_qg_model:
            raise HTTPException(status_code=503, detail="T5 model is not available.")
        
        loop = asyncio.get_running_loop()
        tokenizer = t5_qg_model['tokenizer']
        model = t5_qg_model['model']
        device = t5_qg_model['device']
        
        def _t5_gen():
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
            outputs = model.generate(**inputs, max_length=max_tokens)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        t = await loop.run_in_executor(None, _t5_gen)
    
    elif model_name == "gpt2":
        if not gpt2_pipeline:
            raise HTTPException(status_code=503, detail="GPT-2 model is not available.")
        
        loop = asyncio.get_running_loop()
        
        try:
            result = await loop.run_in_executor(
                None,
                lambda: gpt2_pipeline(
                    prompt,
                    max_length=min(max_tokens + len(prompt.split()), 1024),
                    do_sample=True,
                    temperature=0.7,
                    truncation=True,
                    pad_token_id=50256,
                    num_return_sequences=1
                )
            )
            
            if isinstance(result, list) and len(result) > 0:
                full_text = result[0].get('generated_text', '')
            elif isinstance(result, dict):
                full_text = result.get('generated_text', '')
            else:
                full_text = str(result)
            
            if full_text.startswith(prompt):
                t = full_text[len(prompt):].strip()
            else:
                t = full_text.strip()
        
        except TypeError:
            result = await loop.run_in_executor(
                None,
                lambda: gpt2_pipeline(prompt, max_length=512, do_sample=True, pad_token_id=50256)
            )
            
            if isinstance(result, list) and len(result) > 0:
                full_text = result[0].get('generated_text', '')
            elif isinstance(result, dict):
                full_text = result.get('generated_text', '')
            else:
                full_text = str(result)
            
            if full_text.startswith(prompt):
                t = full_text[len(prompt):].strip()
            else:
                t = full_text.strip()
    
    else:
        if not ollama_client:
            raise HTTPException(status_code=503, detail="Ollama client is not available.")
        
        a = model_name.lower()
        if "gemma" in a or "llama" in a or "qwen" in a:
            prompt = clean_ocr_context_for_generation(prompt)
            b = (
                "IMPORTANT INSTRUCTION: You are generating science quiz questions based on textbook text. "
                "STRICTLY IGNORE any references to figures, images, or diagrams. "
                "STRICTLY IGNORE specific personal names. "
                "CONVERT narrative scenarios into generic scientific questions. "
                "Example: Instead of 'Snehal observed that...', write 'What happens when...'. "
                "Do NOT start questions with 'Observe the set-up'.\n\n"
                f"{prompt}"
            )
            response_data = await ollama_client.generate(model=model_name, prompt=b, stream=False)
        else:
            response_data = await ollama_client.generate(model=model_name, prompt=prompt, stream=False)
        
        t = response_data.get("response", "").strip()
    
    return t

async def _generate_fallback_questions(
    ctx: str,
    q_type: str,
    count: int,
    model_name: str,
    parse_func: Callable[[str], List[Dict[str, Union[str, None]]]]
) -> List[Dict[str, Union[str, None]]]:
    """Generate questions as fallback when primary generation fails"""
    prompt_instruction = ""
    
    if q_type == "mcqs":
        prompt_instruction = f"Generate exactly {count} Multiple Choice Questions (MCQs) with four options (A, B, C, D). Do not include the answer."
    elif q_type == "fill_in_the_blanks":
        prompt_instruction = f"Generate exactly {count} fill-in-the-blanks questions. Replace a key word or phrase with '____'. Do not include the answer."
    elif q_type == "subjective":
        prompt_instruction = f"Generate exactly {count} subjective (long-answer) questions. Do not include the answer."
    else:
        return []

    p = f"""
You are an assistant that generates a specific type of question from the provided context.
Base all questions ONLY on the "Full Context" provided below.

**Full Context:**
---
{ctx}
---

**Task:**
{prompt_instruction}

**Output:**
Provide ONLY the numbered list of questions. Do not include any other text, headers, or explanations.
"""
    
    try:
        t = await _run_llm_call(model_name, p, max_tokens=(count * 150))
        if not t:
            return []
        return parse_func(t)
    except Exception as ex:
        traceback.print_exc()
        return []

# --- API ENDPOINTS ---

@app.post("/extract-text")
async def extract_text_endpoint(file: UploadFile = File(...)):
    """Extract text from uploaded files"""
    print(f"Received file for extraction: {file.filename}")
    
    try:
        content = await file.read()
        file_stream = BytesIO(content)
        extracted = extract_text_from_stream(file_stream, file.filename)
        
        if not extracted or len(extracted.strip()) < 10:
            raise HTTPException(status_code=400, detail="Could not extract readable text.")
        
        return {"extracted_text": extracted, "filename": file.filename}
    
    except Exception as ex:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(ex)}")

@app.post("/generate-questions")
async def generate_questions(
    counts_json: str = Form(...),
    model_name: str = Form("llama3.1"),
    class_num: Optional[str] = Form(None),
    subject: Optional[str] = Form(None),
    custom_context: Optional[str] = Form(None),
    source_filename: Optional[str] = Form(None)  # NEW: Original uploaded filename
):
    """Generate questions based on context and model"""
    ctx = ""
    actual_source_filename = None
    
    if custom_context and len(custom_context.strip()) > 0:
        print("Using Custom Uploaded Context for generation.")
        c = model_name.lower()
        if "gemma" in c or "llama" in c or "qwen" in c:
            ctx = clean_ocr_context_for_generation(custom_context)
        else:
            ctx = custom_context
        actual_source_filename = source_filename  # Store the uploaded filename
    else:
        if not class_num or not subject:
            raise HTTPException(
                status_code=400,
                detail="Please either upload a document OR select a Class and Subject."
            )
        try:
            raw_ctx = get_context_from_selection(class_num, subject)
            d = model_name.lower()
            if "gemma" in d or "llama" in d or "qwen" in d:
                ctx = clean_ocr_context_for_generation(raw_ctx)
            else:
                ctx = raw_ctx
        except HTTPException as ex:
            raise ex

    try:
        counts = QuestionCounts.model_validate_json(counts_json)
    except ValidationError as ex:
        raise HTTPException(status_code=400, detail=f"Invalid JSON for question counts: {ex}")

    output_data = {"mcqs": [], "fill_in_the_blanks": [], "subjective": []}

    try:
        # [Keep all the generation logic the same - T5, GPT-2, and other models]
        # I'm abbreviating here for space, but include ALL the generation code from before
        
        if model_name == "t5":
            # T5 generation logic (same as before)
            sentences = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', ctx.strip()) if len(s.strip()) > 30]
            if not sentences:
                sentences = [ctx]

            async def gen_t5_smart(sentence):
                words = sentence.split()
                keywords = [w for w in words if len(w) > 4 and w.lower() not in ("which", "where", "about", "their", "there")]
                target = keywords[0] if keywords else (words[-1] if words else "it")
                p = f"generate question: <hl> {target} <hl> {sentence}"
                return await _run_llm_call("t5", p, max_tokens=64)

            for i in range(counts.mcqs):
                if not sentences:
                    break
                s_idx = i % len(sentences)
                q = await gen_t5_smart(sentences[s_idx])
                dist1 = sentences[(s_idx + 1) % len(sentences)].split()[0] + "..."
                dist2 = sentences[(s_idx + 2) % len(sentences)].split()[0] + "..."
                output_data["mcqs"].append({
                    "question": f"{q}\nA) {dist1}\nB) {dist2}\nC) Correct Choice\nD) None",
                    "answer": None
                })

            for i in range(counts.fill_in_the_blanks):
                if not sentences:
                    break
                s_idx = (i + 5) % len(sentences)
                original_sent = sentences[s_idx]
                
                words = re.findall(r'\b\w+\b', original_sent)
                candidates = [w for w in words if len(w) > 4 and w.lower() not in
                            ("which", "where", "about", "their", "there", "these", "those",
                             "because", "through", "would", "could")]
                
                if candidates:
                    target = candidates[-1] if len(candidates) > 1 else candidates[0]
                    pattern = re.compile(re.escape(target), re.IGNORECASE)
                    fib_q = pattern.sub("______", original_sent, count=1)
                else:
                    fib_q = re.sub(r'\b\w+[\.\?!]?$', "______.", original_sent)

                output_data["fill_in_the_blanks"].append({"question": fib_q, "answer": None})

            for i in range(counts.subjective):
                if not sentences:
                    break
                s_idx = (i + 3) % len(sentences)
                q = await gen_t5_smart(sentences[s_idx])
                output_data["subjective"].append({"question": q, "answer": None})

        elif model_name == "gpt2":
            # GPT-2 generation (same as before - abbreviated for space)
            chunks = chunk_text_for_gpt2(ctx, approx_chunk_chars=1800)
            if not chunks:
                chunks = [ctx[:4000]]
            
            n_chunks = len(chunks)
            mcq_alloc = [counts.mcqs // n_chunks] * n_chunks
            fib_alloc = [counts.fill_in_the_blanks // n_chunks] * n_chunks
            subj_alloc = [counts.subjective // n_chunks] * n_chunks
            
            for i in range(counts.mcqs % n_chunks):
                mcq_alloc[i] += 1
            for i in range(counts.fill_in_the_blanks % n_chunks):
                fib_alloc[i] += 1
            for i in range(counts.subjective % n_chunks):
                subj_alloc[i] += 1

            agg_mcq_text, agg_fib_text, agg_subj_text = "", "", ""
            
            for idx, chunk in enumerate(chunks):
                sub_prompts = []
                if mcq_alloc[idx] > 0:
                    sub_prompts.append(f"Generate {mcq_alloc[idx]} MCQs. List 1. ...")
                if fib_alloc[idx] > 0:
                    sub_prompts.append(f"Generate {fib_alloc[idx]} blanks.")
                if subj_alloc[idx] > 0:
                    sub_prompts.append(f"Generate {subj_alloc[idx]} subjective questions.")
                
                if not sub_prompts:
                    continue
                
                prompt_chunk = "\n".join(sub_prompts) + "\n" + chunk
                raw_out = await _run_llm_call("gpt2", prompt_chunk, max_tokens=512)
                cleaned = sanitize_chunk_output(raw_out)
                
                agg_mcq_text += "\n" + cleaned
                agg_fib_text += "\n" + cleaned
                agg_subj_text += "\n" + cleaned

            output_data["mcqs"] = parse_questions_from_block(agg_mcq_text)
            output_data["fill_in_the_blanks"] = parse_questions_from_block(agg_fib_text)
            output_data["subjective"] = parse_questions_from_block(agg_subj_text)
            
            if not output_data["mcqs"] and counts.mcqs > 0:
                output_data["mcqs"] = await _generate_fallback_questions(
                    ctx, "mcqs", counts.mcqs, "gpt2", parse_questions_from_block
                )
            if not output_data["fill_in_the_blanks"] and counts.fill_in_the_blanks > 0:
                output_data["fill_in_the_blanks"] = await _generate_fallback_questions(
                    ctx, "fill_in_the_blanks", counts.fill_in_the_blanks, "gpt2", parse_questions_from_block
                )
            if not output_data["subjective"] and counts.subjective > 0:
                output_data["subjective"] = await _generate_fallback_questions(
                    ctx, "subjective", counts.subjective, "gpt2", parse_questions_from_block
                )

        else:
            # OTHER MODELS (Same as before)
            if "phi3-rank4" in model_name.lower():
                a = 3000
                b = ctx[:a] if len(ctx) > a else ctx

                c = f"""You are a question generator. Use the "Context" below.
Task: Generate exactly {counts.mcqs} Multiple Choice Questions (MCQs).
Format Requirements:
- Number each question (1., 2., etc.)
- Provide 4 distinct options (A, B, C, D) on separate lines.
- Do NOT include the correct answer.

Context:
{b}
"""

                d = f"""You are a question generator. Use the "Context" below.
Task: Generate exactly {counts.fill_in_the_blanks} Fill-in-the-blank sentences.
Format Requirements:
- Number each sentence (1., 2., etc.)
- Use '____' to represent the missing word.
- Do NOT provide the answers.

Context:
{b}
"""

                e = f"""You are a question generator. Use the "Context" below.
Task: Generate exactly {counts.subjective} Subjective (long-answer) questions.
Format Requirements:
- Number each question (1., 2., etc.)
- Do NOT provide the answers.

Context:
{b}
"""

                f, g, h = await asyncio.gather(
                    _run_llm_call(model_name, c, max_tokens=800),
                    _run_llm_call(model_name, d, max_tokens=600),
                    _run_llm_call(model_name, e, max_tokens=600),
                )

                i = f"Part 1: Multiple Choice Questions\n{f}\n\nPart 2: Fill in the Blanks\n{g}\n\nPart 3: Subjective Questions\n{h}"

            else:
                p = f"""
You are an assistant that generates a three-part quiz based on the provided text.
Base all questions ONLY on the "Full Context" provided below.

**Full Context:**
---
{ctx}
---

Follow these instructions exactly.

**Part 1: Multiple Choice Questions**
Generate exactly {counts.mcqs} MCQs. **Do NOT include the answer.**

**Part 2: Fill in the Blanks**
Generate exactly {counts.fill_in_the_blanks} fill-in-the-blanks questions. **Do NOT include the answer.**

**Part 3: Subjective Questions**
Generate exactly {counts.subjective} subjective (long-answer) questions. **Do NOT include the answer.**
"""
                t = await _run_llm_call(model_name, p, max_tokens=2048)
            
            def get_part_start_index(text, part_num):
                pattern = r'(?:^|\n)\s*(?:\*\*)?\s*Part\s*' + str(part_num)
                match = re.search(pattern, text, re.IGNORECASE)
                return match.start() if match else -1

            idx_1 = get_part_start_index(t, 1)
            idx_2 = get_part_start_index(t, 2)
            idx_3 = get_part_start_index(t, 3)
            total_len = len(t)
            
            mcq_content = t[idx_1:(idx_2 if idx_2 != -1 else (idx_3 if idx_3 != -1 else total_len))] if idx_1 != -1 else ""
            fib_content = t[idx_2:(idx_3 if idx_3 != -1 else total_len)] if idx_2 != -1 else ""
            sub_content = t[idx_3:] if idx_3 != -1 else ""

            output_data["mcqs"] = parse_questions_from_block(mcq_content)
            output_data["fill_in_the_blanks"] = parse_questions_from_block(fib_content)
            output_data["subjective"] = parse_questions_from_block(sub_content)

            if not output_data["mcqs"] and counts.mcqs > 0:
                output_data["mcqs"] = await _generate_fallback_questions(
                    ctx, "mcqs", counts.mcqs, model_name, parse_questions_from_block
                )
            if not output_data["fill_in_the_blanks"] and counts.fill_in_the_blanks > 0:
                output_data["fill_in_the_blanks"] = await _generate_fallback_questions(
                    ctx, "fill_in_the_blanks", counts.fill_in_the_blanks, model_name, parse_questions_from_block
                )
            if not output_data["subjective"] and counts.subjective > 0:
                output_data["subjective"] = await _generate_fallback_questions(
                    ctx, "subjective", counts.subjective, model_name, parse_questions_from_block
                )

        # NEW: Save the generated paper to disk
        saved_path = save_generated_paper(
            questions=output_data,
            context=ctx,
            model_name=model_name,
            class_num=class_num,
            subject=subject,
            source_filename=actual_source_filename
        )
        
        print(f"✅ Question paper saved: {saved_path}")

        return {
            "questions": output_data,
            "context": ctx,
            "saved_filepath": saved_path  # Return the saved file path to frontend
        }

    except Exception as ex:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Question generation failed: {ex}")

# [Continue with all other endpoints - evaluate-accuracy, download-pdf, submit-ratings, etc.]
# Keep them exactly the same as before

@app.post("/evaluate-accuracy")
async def evaluate_accuracy(
    question: str = Form(...),
    generated_answer: str = Form(...),
    q_type: str = Form(...),
    metric: str = Form("bert"),
    class_num: Optional[str] = Form(None),
    subject: Optional[str] = Form(None),
    custom_context: Optional[str] = Form(None),
    reference_file: UploadFile = File(None)
):
    """Evaluate generated question quality by comparing to reference questions"""
    print(f"🔍 Evaluating QUESTION quality: class={class_num}, subject={subject}, metric={metric}")
    
    reference_questions = []
    
    if reference_file:
        print(f"📄 Using uploaded reference file: {reference_file.filename}")
        try:
            content = await reference_file.read()
            file_stream = BytesIO(content)
            extracted_text = extract_text_from_stream(file_stream, reference_file.filename)
            reference_questions = parse_reference_questions_from_text(extracted_text)
            print(f"✓ Parsed {len(reference_questions)} reference questions from uploaded file.")
        except Exception as ex:
            print(f"✗ Error processing uploaded reference file: {ex}")
            traceback.print_exc()
    
    elif class_num and class_num != "Custom":
        reference_questions = load_reference_questions(class_num, subject)
        print(f"📚 Loaded {len(reference_questions)} reference questions from directory.")
    
    if not reference_questions:
        return {
            "generated_answer": "No reference questions available for comparison",
            "bert_answer": "❌ No reference questions found. Please upload a reference file or ensure textbook questions are available.",
            "similarity_score": 0.0
        }
    
    clean_gen_q = clean_question_for_comparison(question)
    print(f"📝 Generated Question (cleaned): {clean_gen_q[:100]}...")
    
    match_result = find_most_similar_reference_question(clean_gen_q, reference_questions, similarity_threshold=0.20)
    
    if not match_result:
        return {
            "generated_answer": "No similar reference question found",
            "bert_answer": "⚠️ No matching reference question found (similarity too low)",
            "similarity_score": 0.0
        }
    
    reference_question = match_result['question']
    clean_ref_q = clean_question_for_comparison(reference_question)
    
    print(f"📚 Reference Question (cleaned): {clean_ref_q[:100]}...")
    
    score = 0.0
    
    try:
        if metric == "bleu":
            ref_tokens = [nltk.word_tokenize(clean_ref_q.lower())]
            cand_tokens = nltk.word_tokenize(clean_gen_q.lower())
            
            score = sentence_bleu(
                ref_tokens,
                cand_tokens,
                smoothing_function=SmoothingFunction().method1
            )
            print(f"📊 BLEU Score: {score:.4f}")

        elif metric == "meteor":

            ref_tokens = nltk.word_tokenize(clean_ref_q.lower())
            cand_tokens = nltk.word_tokenize(clean_gen_q.lower())

            score = meteor_score([ref_tokens], cand_tokens)

            print(f"📊 METEOR Score: {score:.4f}")
        
        elif metric == "rouge":
            scores = rouge_evaluator.score(clean_ref_q, clean_gen_q)
            score = scores['rougeL'].fmeasure
            print(f"📊 ROUGE-L F1 Score: {score:.4f}")
        
        elif metric == "bert" and sentence_transformer:
            emb_ref = sentence_transformer.encode(clean_ref_q, convert_to_tensor=True)
            emb_gen = sentence_transformer.encode(clean_gen_q, convert_to_tensor=True)
            
            sim = util.pytorch_cos_sim(emb_ref, emb_gen)
            score = float(sim.item())
            print(f"📊 BERT Cosine Similarity: {score:.4f}")
        
        else:
            score = SequenceMatcher(None, clean_ref_q.lower(), clean_gen_q.lower()).ratio()
            print(f"📊 String Similarity: {score:.4f}")
        
        display_text = f"Reference Question: {reference_question[:200]}..."
        
        return {
            "generated_answer": f"Compared with textbook question using {metric.upper()}",
            "bert_answer": display_text,
            "similarity_score": score
        }
    
    except Exception as ex:
        print(f"❌ Error during evaluation: {ex}")
        traceback.print_exc()
        
        return {
            "generated_answer": "Evaluation Error",
            "bert_answer": f"Error: {str(ex)}",
            "similarity_score": 0.0
        }

@app.post("/download-questions-pdf")
async def download_questions_pdf(req: PdfRequest):
    """Generate PDF containing only questions"""
    data = req.answered_questions
    
    if not data or not any(data.values()):
        raise HTTPException(status_code=400, detail="No questions provided.")
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    try:
        story.append(Paragraph("Generated Question Paper", styles['h1']))
        story.append(Spacer(1, 12))
        
        if req.model_name:
            story.append(Paragraph(f"<b>Generated by Model:</b> {req.model_name}", styles['Normal']))
            story.append(Spacer(1, 12))
        
        for q_type, q_list in data.items():
            if q_list:
                header = q_type.replace('_', ' ').title()
                story.append(Paragraph(header, styles['h2']))
                
                for i, item in enumerate(q_list, 1):
                    q_text_safe = (item.question or "No Question").replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    story.append(Paragraph(f"{i}. {q_text_safe}".replace('\n', '<br/>'), styles['Normal']))
                    story.append(Spacer(1, 24))
        
        doc.build(story)
        buffer.seek(0)
    
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to generate Question Paper PDF: {ex}")
    
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=question_paper.pdf"}
    )

@app.post("/submit-difficulty-ratings")
async def submit_difficulty_ratings(ratings_data: DifficultyRatings):
    """Submit expert difficulty ratings"""
    print("--- Received Difficulty Ratings ---")
    print(f"Class: {ratings_data.class_num}, Subject: {ratings_data.subject}")
    print(f"Ratings Data: {ratings_data.model_dump_json(indent=2)}")
    
    return {"status": "success", "message": "Ratings logged successfully."}

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Quiz Generator API is running",
        "ollama_client_status": "Initialized" if ollama_client else "Failed",
        "openai_client_status": "Initialized" if openai_client else "Failed (Check API Key)",
        "qa_model_status": "Loaded" if qa_pipeline else "Failed",
        "sentence_transformer_status": "Loaded" if sentence_transformer else "Failed",
        "summarizer_status": "Loaded" if summarizer_pipeline else "Failed",
        "t5_qg_status": "Loaded" if t5_qg_model else "Failed",
        "gpt2_gen_status": "Loaded" if gpt2_pipeline else "Failed",
        "ocr_status": "Available"
    }

@app.get("/check-models")
async def check_models():
    """Check status of all models"""
    ollama_available = False
    models_list = []
    
    if ollama_client:
        try:
            list_response = await ollama_client.list()
            ollama_available = True
            models_list = [m['name'] for m in list_response.get('models', [])]
        except Exception as ex:
            print(f"Could not reach Ollama service: {ex}")
            ollama_available = False
    
    return {
        "ollama_client": "Initialized" if ollama_client else "Failed",
        "ollama_service_reachable": ollama_available,
        "ollama_models_available": models_list,
        "openai_client": "Initialized" if openai_client else "Failed (Check API Key)",
        "qa_model_status": "Loaded" if qa_pipeline else "Failed",
        "sentence_transformer": "Loaded" if sentence_transformer else "Failed",
        "summarizer": "Loaded" if summarizer_pipeline else "Failed",
        "t5_qg_model": "Loaded" if t5_qg_model else "Failed",
        "gpt2_gen_model": "Loaded" if gpt2_pipeline else "Failed"
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)