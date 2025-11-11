# Server Code (FullServer.py)
from fastapi import FastAPI, APIRouter, Body, HTTPException, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from memory_manager import MemoryManager
from chat_llm_service import ChatLLMService
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
import json
import uuid
import os
import logging
from contextlib import asynccontextmanager
import re  # For password validation

# --- CONFIGURATION ---
DB_URL = "sqlite:///data/chatbot.db"
SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key-replace-with-env-var-in-prod")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # One day

MODEL_DIR = "models"
MODEL_FILENAME = "PhysicsChatbot.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

os.makedirs(MODEL_DIR, exist_ok=True)

GREETINGS = {
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "morning", "afternoon", "evening", "hola", "salutations", "greetings",
    "thanks", "tanx", "thank you", "no", "yes"
}

# --- Initialize logger ---
app_logger_instance = TechSupportLogger(
    log_file_name="middleware.log",
    log_dir="data/logs",
    level=logging.DEBUG,
    max_bytes=50 * 1024 * 1024,
    backup_count=10,
    console_output=True
)
logger = app_logger_instance.get_logger()

# --- DATABASE SETUP ---
try:
    engine = create_engine(DB_URL)
except Exception as e:
    logger.critical(f"Failed to create database engine: {e}", exc_info=True)
    raise
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

memory_store = {}

# --- LLM Initialization ---
try:
    llm_service = ChatLLMService(model_path=MODEL_PATH)
    logger.info(f"ChatLLMService initialized with model: {MODEL_PATH}")
except Exception as e:
    logger.critical(f"Failed to initialize ChatLLMService: {e}", exc_info=True)
    llm_service = None
    raise

# --- FastAPI Lifespan Events ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI application startup initiated.")
    yield
    logger.info("FastAPI application shutdown initiated.")
    for key, mem_manager in list(memory_store.items()):
        try:
            logger.info(f"Saving memory for conversation: {key}")
            mem_manager._safe_save()
            del memory_store[key]
        except Exception as e:
            logger.error(f"Failed to save memory for {key}: {e}", exc_info=True)
    logger.info("All active memory managers saved. Shutdown complete.")

# --- APP SETUP ---
app = FastAPI(lifespan=lifespan)
chat_router = APIRouter()

def init_db():
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialized.")

init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust for JavaFX client
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/llm/token")

# --- UTILS ---
def get_db():
    try:
        db = SessionLocal()
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database connection failed")
    finally:
        db.close()

# --- MODELS ---
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    chats = relationship("Chat", back_populates="user")

class Chat(Base):
    __tablename__ = 'chats'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="chats")
    conversation_id = Column(String(36), index=True)

class PasswordUpdateRequest(BaseModel):
    old_password: str
    new_password: str

# --- AUTHENTICATION FUNCTIONS ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def validate_password(password: str) -> bool:
    if len(password) < 8:
        return False
    if not re.search(r"[A-Z]", password) or not re.search(r"[0-9]", password):
        return False
    return True

# --- AUTH ROUTES ---
@chat_router.post("/register", summary="Register a new user")
def register_user(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    logger.info(f"Attempting to register user: {form_data.username}")
    try:
        if not validate_password(form_data.password):
            logger.warning(f"Registration failed for '{form_data.username}': Password too weak")
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters with uppercase and numbers")
        if db.query(User).filter_by(username=form_data.username).first():
            logger.warning(f"Registration failed: Username '{form_data.username}' already exists")
            raise HTTPException(status_code=400, detail="Username already exists")
        hashed = get_password_hash(form_data.password)
        user = User(username=form_data.username, hashed_password=hashed)
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"User '{form_data.username}' registered successfully with ID: {user.id}")
        return {"msg": "User registered successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during user registration for '{form_data.username}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during registration")

@chat_router.post("/token", summary="Authenticate user and get access token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    logger.info(f"Attempting to log in user: {form_data.username}")
    try:
        user = db.query(User).filter_by(username=form_data.username).first()
        if not user or not verify_password(form_data.password, user.hashed_password):
            logger.warning(f"Login failed for user '{form_data.username}': Invalid credentials")
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = create_access_token({"sub": user.username})
        logger.info(f"User '{form_data.username}' logged in successfully")
        return {"access_token": token, "token_type": "bearer", "username": user.username}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during user login for '{form_data.username}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during login")

@chat_router.post("/update-password", summary="Update user password")
def update_password(
    password_data: PasswordUpdateRequest,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    logger.info("Password update request received.")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except JWTError as e:
        logger.error(f"JWT error during password update: {e}", exc_info=True)
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter_by(username=username).first()
    if not user:
        logger.warning(f"Password update failed: User '{username}' not found")
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(password_data.old_password, user.hashed_password):
        logger.warning(f"Password update failed for '{username}': Incorrect old password")
        raise HTTPException(status_code=400, detail="Incorrect old password")

    if not validate_password(password_data.new_password):
        logger.warning(f"Password update failed for '{username}': New password too weak")
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters with uppercase and numbers")

    user.hashed_password = get_password_hash(password_data.new_password)
    db.commit()
    logger.info(f"Password updated successfully for user '{username}'")
    return {"msg": "Password updated successfully"}

@chat_router.post("/chat", summary="Stream chat response")
async def chat(
    message: str = Body(...),
    conversation_id: str = Body(None),
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    logger.info(f"Chat request received: {message[:50]}...")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except JWTError as e:
        logger.error(f"JWT error during chat request: {e}", exc_info=True)
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter_by(username=username).first()
    if not user:
        logger.warning(f"Chat failed: User '{username}' not found")
        raise HTTPException(status_code=404, detail="User not found")

    key = f"{username}:{conversation_id}" if conversation_id else f"{username}:{str(uuid.uuid4())}"
    conversation_id = key.split(":")[1] if not conversation_id else conversation_id

    if key not in memory_store:
        memory_dir = os.path.join("data", "memory", username, conversation_id)
        os.makedirs(memory_dir, exist_ok=True)
        memory_path = os.path.join(memory_dir, "memory.pkl")
        memory_store[key] = MemoryManager(memory_path=memory_path)
        logger.info(f"Created new MemoryManager for {key}")

    memory = memory_store[key]
    context = memory.get_context(message)
    formatted_prompt = llm_service.format_prompt(message, context, [])  # No chunks

    async def stream_and_store():
        assistant_reply = ""
        yield json.dumps({"response": "", "conversation_id": conversation_id}) + "\n"
        try:
            for token in llm_service.generate_response_stream(formatted_prompt):
                assistant_reply += token
                yield json.dumps({"response": token, "conversation_id": conversation_id}) + "\n"
            memory.add_message("assistant", assistant_reply)
            db.add(Chat(user_id=user.id, conversation_id=conversation_id, role="user", content=message))
            db.add(Chat(user_id=user.id, conversation_id=conversation_id, role="assistant", content=assistant_reply))
            db.commit()
            logger.info(f"Streamed response saved for conversation '{conversation_id}'")
        except Exception as e:
            logger.error(f"LLM streaming error for '{key}': {e}", exc_info=True)
            yield json.dumps({"error": "Streaming failed", "conversation_id": conversation_id}) + "\n"

    return StreamingResponse(stream_and_store(), media_type="application/json", headers={"X-Conversation-ID": conversation_id})

@chat_router.get("/conversations", summary="List all user conversations")
async def list_conversations(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except JWTError as e:
        logger.error(f"JWT error during list_conversations request: {e}", exc_info=True)
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter_by(username=username).first()
    if not user:
        logger.warning(f"List conversations failed: User '{username}' not found")
        raise HTTPException(status_code=404, detail="User not found")

    conversation_ids_query = (
        db.query(Chat.conversation_id)
        .filter_by(user_id=user.id)
        .distinct()
        .all()
    )
    conversation_ids = [conv_id_tuple[0] for conv_id_tuple in conversation_ids_query]
    logger.info(f"User '{username}' requested conversations. Found {len(conversation_ids)} conversations")

    conversations = []
    for conv_id in conversation_ids:
        first_user_msg = (
            db.query(Chat)
            .filter_by(user_id=user.id, conversation_id=conv_id, role="user")
            .order_by(Chat.timestamp.asc())
            .first()
        )
        first_assistant_msg = (
            db.query(Chat)
            .filter_by(user_id=user.id, conversation_id=conv_id, role="assistant")
            .order_by(Chat.timestamp.asc())
            .first()
        )

        header = first_user_msg.content if first_user_msg else "No user message"
        description = first_assistant_msg.content if first_assistant_msg else "No assistant message"

        conversations.append({
            "conversation_id": conv_id,
            "conversation_header": " ".join(header.split()[:20]),
            "conversation_desc": " ".join(description.split()[:20])
        })
    return {"messages": conversations, "total": len(conversations)}

@chat_router.get("/conversations/{conversation_id}", summary="Get messages for a specific conversation")
async def get_conversation(conversation_id: str, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    logger.info(f"Fetching conversation '{conversation_id}' for user")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except JWTError as e:
        logger.error(f"JWT error during get_conversation request: {e}", exc_info=True)
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter_by(username=username).first()
    if not user:
        logger.warning(f"Get conversation failed: User '{username}' not found")
        raise HTTPException(status_code=404, detail="User not found")

    chats = db.query(Chat).filter_by(user_id=user.id, conversation_id=conversation_id).order_by(Chat.timestamp).all()
    if not chats:
        logger.warning(f"Conversation '{conversation_id}' not found for user '{username}'")
        raise HTTPException(status_code=404, detail="Conversation not found")

    user_msgs = [chat.content for chat in chats if chat.role == "user"]
    bot_msgs = [chat.content for chat in chats if chat.role == "assistant"]
    logger.info(f"Retrieved {len(user_msgs)} user messages and {len(bot_msgs)} bot messages for conversation '{conversation_id}'")
    return {
        "conversation_id": conversation_id,
        "conversations": {
            "userMessages": user_msgs,
            "botMessages": bot_msgs
        }
    }

@chat_router.delete("/conversations/{conversation_id}", summary="Delete a conversation")
async def delete_conversation(conversation_id: str, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    logger.info(f"Attempting to delete conversation '{conversation_id}'")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except JWTError as e:
        logger.error(f"JWT error during delete_conversation request: {e}", exc_info=True)
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter_by(username=username).first()
    if not user:
        logger.warning(f"Delete conversation failed: User '{username}' not found")
        raise HTTPException(status_code=404, detail="User not found")

    chats = db.query(Chat).filter_by(user_id=user.id, conversation_id=conversation_id).all()
    if not chats:
        logger.warning(f"Conversation '{conversation_id}' not found for user '{username}'")
        raise HTTPException(status_code=404, detail="Conversation not found")

    for chat in chats:
        db.delete(chat)
    db.commit()
    logger.info(f"Conversation '{conversation_id}' deleted from DB for user '{username}'")

    key = f"{username}:{conversation_id}"
    memory_dir = os.path.join("data", "memory", username, conversation_id)
    if os.path.exists(memory_dir):
        try:
            shutil.rmtree(memory_dir)
            logger.info(f"Deleted memory directory: {memory_dir}")
        except Exception as e:
            logger.error(f"Failed to delete memory directory {memory_dir}: {e}", exc_info=True)

    if key in memory_store:
        try:
            memory_store[key]._safe_save()
            del memory_store[key]
            logger.info(f"Removed MemoryManager for {key}")
        except Exception as e:
            logger.error(f"Failed to save/delete MemoryManager for {key}: {e}", exc_info=True)

    return {"status": "deleted", "conversation_id": conversation_id}

# --- ROUTE REGISTRATION ---
app.include_router(chat_router, prefix="/llm")

@app.get("/", summary="Health check endpoint")
def root():
    logger.info("Health check request received")
    return {"status": "running"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
