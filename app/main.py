import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Form, UploadFile, File, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import UploadFile as StarletteUploadFile

from dotenv import load_dotenv
from app.utils import (
    load_document, summarize_document, create_vectorstore, answer_query, MODEL_DIR
)

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import sys, asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / 'templates'))
app.mount('/static', StaticFiles(directory=str(BASE_DIR / 'static')), name='static')

# In-memory stores (for prototype). For production use persistent DB or per-user storage.
SESSIONS = {}


def get_session(request: Request):
    session_id = request.cookies.get('session_id')
    if not session_id:
        session_id = uuid.uuid4().hex
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            'uploaded_path': None,
            'extracted_text': None,
            'summary': None,
            'vector_meta': None,
            'index_path': None,
            'chat_history': [],
            'is_summarizing': False,# list of (user, assistant)
        }
    return session_id, SESSIONS[session_id]


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    session_id, session = get_session(request)
    response = TEMPLATES.TemplateResponse('index.html', {
        'request': request,
        'extracted_text': session['extracted_text'],
        'summary': session['summary'],
        'chat_history': session['chat_history'],
    })
    # set cookie if not present
    if not request.cookies.get('session_id'):
        response.set_cookie('session_id', session_id)
    return response


@app.post('/upload')
async def upload(request: Request, file: UploadFile = File(...)):
    session_id, session = get_session(request)
    UPLOAD_DIR = BASE_DIR / 'uploads'
    UPLOAD_DIR.mkdir(exist_ok=True)

    file_ext = Path(file.filename).suffix.lower()
    saved = UPLOAD_DIR / f"{session_id}{file_ext}"
    with open(saved, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    # extract text
    try:
        text = load_document(str(saved))
    except Exception as e:
        return TEMPLATES.TemplateResponse('index.html', {
            'request': request,
            'error': f'Failed to extract text: {e}',
            'extracted_text': None,
            'summary': None,
            'chat_history': session['chat_history'],
        })

    session['uploaded_path'] = str(saved)
    session['extracted_text'] = text
    session['summary'] = None
    session['vector_meta'] = None
    session['index_path'] = None

    return RedirectResponse(url='/', status_code=303)


@app.post('/summarize')
async def summarize(request: Request):
    session_id, session = get_session(request)
    if not session['extracted_text']:
        return TEMPLATES.TemplateResponse('index.html', {
            'request': request,
            'error': 'No document uploaded.',
            'extracted_text': None,
            'summary': None,
            'chat_history': session['chat_history'],
        })

    if session.get('is_summarizing'):
        return TEMPLATES.TemplateResponse('index.html', {
            'request': request,
            'error': 'Summarization already in progress…',
            'extracted_text': session['extracted_text'],
            'summary': session['summary'],
            'chat_history': session['chat_history'],
        })

    session['is_summarizing'] = True
    try:
        summary = summarize_document(session['extracted_text'])
        session['summary'] = summary
    except Exception as e:
        return TEMPLATES.TemplateResponse('index.html', {
            'request': request,
            'error': f'Summarization failed: {e}',
            'extracted_text': session['extracted_text'],
            'summary': None,
            'chat_history': session['chat_history'],
        })
    finally:
        session['is_summarizing'] = False

    return RedirectResponse(url='/', status_code=303)


@app.post('/create_vectorstore')
async def create_vectorstore_route(request: Request):
    session_id, session = get_session(request)
    if not session['extracted_text']:
        return TEMPLATES.TemplateResponse('index.html', {
            'request': request,
            'error': 'No document uploaded.',
            'extracted_text': None,
            'summary': None,
            'chat_history': session['chat_history'],
        })

    try:
        index_path, meta = create_vectorstore(session['extracted_text'])
        session['index_path'] = index_path
        session['vector_meta'] = meta
    except Exception as e:
        return TEMPLATES.TemplateResponse('index.html', {
            'request': request,
            'error': f'Vector store creation failed: {e}',
            'extracted_text': session['extracted_text'],
            'summary': session['summary'],
            'chat_history': session['chat_history'],
        })

    return RedirectResponse(url='/', status_code=303)


@app.post('/chat')
async def chat(request: Request, question: str = Form(...)):
    session_id, session = get_session(request)
    if not session.get('index_path') or not session.get('vector_meta'):
        return TEMPLATES.TemplateResponse('index.html', {
            'request': request,
            'error': 'Please create a vectorstore first (create vectorstore button).',
            'extracted_text': session['extracted_text'],
            'summary': session['summary'],
            'chat_history': session['chat_history'],
        })

    try:
        assistant_text, extra = answer_query(session['index_path'], session['vector_meta'], question, conversation_history=session['chat_history'])
        # append to history
        session['chat_history'].append((question, assistant_text))
    except Exception as e:
        return TEMPLATES.TemplateResponse('index.html', {
            'request': request,
            'error': f'Chat failed: {e}',
            'extracted_text': session['extracted_text'],
            'summary': session['summary'],
            'chat_history': session['chat_history'],
        })

    return RedirectResponse(url='/', status_code=303)