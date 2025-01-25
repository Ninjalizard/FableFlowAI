from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from openai import AsyncOpenAI, OpenAIError

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use environment variable for API key, or replace with your key directly
# Note, replace "your-api-key-here" with os.environ.get("OPENAI_API_KEY")
#client = AsyncOpenAI(api_key="<PUT OPENAI KEY HERE>")
client = AsyncOpenAI(api_key="")
class StoryRequest(BaseModel):
    prompt: Optional[str]
    genre: str
    style: str
    type: str
    setting: str
    previous_paragraphs: List[str] = []
    user_input: Optional[str] = None

@app.post("/generate-paragraph")
async def generate_paragraph(request: StoryRequest):
    try:
        # Build the context from previous paragraphs
        story_context = "\n\n".join(request.previous_paragraphs)

        # Create system prompt with story parameters
        system_prompt = f"You are a creative writer specializing in {request.genre} stories. Write in a {request.style} style, following the {request.type} story structure. You tell stories about 
        things that take place in a {request.setting}."

        # Create user prompt based on whether this is the first paragraph or a continuation
        if not request.previous_paragraphs:
            user_prompt = f"Write the opening paragraph for a story about: {request.prompt}"
        else:
            user_prompt = f"""Previous story so far:

{story_context}

User's suggestion for what happens next: {request.user_input}

Write exactly one paragraph continuing the story, incorporating the user's suggestion naturally while maintaining the established {request.style} style and {request.genre} genre elements."""

        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        new_paragraph = response.choices[0].message.content.strip()

        return {
            "paragraph": new_paragraph,
            "status": "success"
        }
    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "Story Generator API is running"}
