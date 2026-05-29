import asyncio
import asyncpg
import uuid
import os
import itertools
import httpx
from typing import TypedDict, Annotated, Literal, cast, Optional
from contextlib import asynccontextmanager
from placement import placement_Retriever
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from qdrant import RulesRetriever
from orcr import ORCR_Retriever
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from user_crud_asyncpg import init_db_pool, close_db_pool, get_pool, get_by_email, create_user, update_user, create_schema, usage_schema, upadate_schema, Category, Gender, get_by_id
from datetime import datetime
from pydantic import BaseModel

load_dotenv()
PASTE_GROQ_KEYS_HERE = os.getenv("GROQ_API_KEY", "")  # comma separated if multiple
PASTE_SERPER_KEY_HERE = os.getenv("SERPER_API_KEY", "")

SESSION_STORAGE = {}

class APIKeyRotator:
    def __init__(self, raw_keys_string: str):
        self.keys = [k.strip() for k in raw_keys_string.split(",") if k.strip()]
        if not self.keys:
            raise ValueError("CRITICAL: No valid API keys provided in configuration.")
        self.pool = itertools.cycle(self.keys)
        print(f"Successfully initialized Key Pool with {len(self.keys)} keys.")

    def get_next_key(self) -> str:
        return next(self.pool)

groq_key_rotator = APIKeyRotator(PASTE_GROQ_KEYS_HERE)

agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing database pool.....")
    await init_db_pool()
    global agent
    agent = OrchestratorAgent(window_size=5)
    agent.initialize()
    yield
    print("Closing database pool...")
    await close_db_pool()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db_retriever = ORCR_Retriever()
retriever = RulesRetriever()
placement_retriever = placement_Retriever()

# ---------- TOOLS WITH DETAILED LOGS ----------
@tool
async def retrieve_college_allocations_JEE_Adv(rank: int, category: str, gender: str) -> str:
    """**CRITICAL: You MUST use this tool for any JEE Advanced rank-based college prediction.**
    Returns a list of actual engineering colleges (IITs) and their opening/closing ranks from the official JoSAA database.
    Do NOT answer from your own knowledge – only use the data returned by this tool.
    gender is strictly : Gender-neutral and Female only. Category is strictly : "OPEN","OBC-NCL","GEN-EWS","SC","ST" .
    """
    print(f"\n🔧 [TOOL] retrieve_college_allocations_JEE_Adv called with rank={rank}, category={category}, gender={gender}")
    try:
        results = await db_retriever.runa(rank, category, gender)
        print(f"   -> DB returned {len(results)} rows")
        if results:
            print(f"   -> First row: {results[0]}")
        if not results:
            return "No colleges found"
        return "Database Matching Allocations:\n" + str(results)
    except Exception as e:
        print(f"    Exception: {e}")
        return f"Database lookup failed: {str(e)}"

@tool
async def retrieve_college_allocations_JEE_Main(rank: int, category: str, gender: str) -> str:
    """**CRITICAL: You MUST use this tool for any JEE Main rank-based college prediction.**
    Returns actual NIT/IIIT allocations from the official JoSAA database.
    Do NOT answer from your own knowledge.
    gender is strictly : Gender-Neutral and Female only. Category is strictly : "OPEN","OBC-NCL","GEN-EWS","SC","ST" .
    """
    print(f"\n🔧 [TOOL] retrieve_college_allocations_JEE_Main called with rank={rank}, category={category}, gender={gender}")
    try:
        results = await db_retriever.runm(rank, category, gender)
        print(f"   -> DB returned {len(results)} rows")
        if results:
            print(f"   -> First row: {results[0]}")
        if not results:
            return "No colleges found"
        return "Database Matching Allocations:\n" + str(results)
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return f"Database lookup failed: {str(e)}"

@tool
async def placement_data(institute: str) -> str:
    """Fetches the latest placement statistics for a specified institute from the database."""
    print(f"\n🔧 [TOOL] placement_data called with institute={institute}")
    try:
        results = await placement_retriever.run(institute)
        print(f"   -> DB returned {len(results)} records")
        if results:
            print(f"   -> First record: {results[0]}")
        if not results:
            return f"No placements data found for {institute}."
        # Build a readable string (keeping original simple format)
        s = ""
        for item in results:
            s += str(item) + "\n"
        return f"Placements results for {institute} are : {s}"
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return f"Placement data retrieval failed."

@tool
async def search_jossa(query: str) -> str:
    """Queries the local indexed document cache for official JoSAA rules, requirements, and reference PDFs."""
    print(f"\n🔧 [TOOL] search_jossa called with query={query}")
    try:
        response = await asyncio.to_thread(retriever.search, query, 3)
        print(f"   -> Retrieved chunks are 3 ")
        if response:
            print(f"   -> First chunk preview: ...")
        return f"Found response for '{query}':\nresponse : {response}"
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return f"JoSAA local index search failed: {str(e)}"

@tool
async def search_google_images(query: str) -> str:
    """Searches Google Images using Serper.dev and returns the top image URL."""
    print(f"\n🔧 [TOOL] search_google_images called with query={query}")
    url = "https://google.serper.dev/images"
    payload = {"q": query, "num": 3}
    headers = {"X-API-KEY": PASTE_SERPER_KEY_HERE, "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            images = data.get("images", [])
            print(f"   -> Found {len(images)} images")
            if images:
                print(f"   -> First image URL: {images[0].get('imageUrl', 'N/A')}")
            if not images:
                return f"No images found for: {query}"
            fi = images[0]
            return f"Image for '{query}' is:\nTitle: {fi.get('title', 'Image')}\nURL: {fi.get('imageUrl')}"
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            return f"Image search failed for query: {query}"

@tool
async def search_web_serper(query: str) -> str:
    """Searches the web via live Google engines to retrieve the latest real-time status and information updates."""
    print(f"\n🔧 [TOOL] search_web_serper called with query={query}")
    if not PASTE_SERPER_KEY_HERE:
        return "Error: SERPER_API_KEY is not set."
    url = "https://google.serper.dev/search"
    payload = {"q": query}
    headers = {"X-API-KEY": PASTE_SERPER_KEY_HERE, "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            snippets = []
            for item in data.get("organic", [])[:3]:
                snippets.append(f"title : {item.get('title')}\nsnippet : {item.get('snippet')}")
            print(f"   -> Got {len(snippets)} snippets")
            if snippets:
                print(f"   -> First snippet: {snippets[0][:100]}...")
            if not snippets:
                return f"Found nothing on web about {query}"
            return "Web search has given:\n" + "\n".join(snippets)
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            return f"Web search for query: {query} returned an error or empty context."

# ---------- AgentState and OrchestratorAgent ----------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    short_term_memory: list
    session_id: str
    user_id: int
    current_input: str
    summary: str

class OrchestratorAgent:
    def __init__(self, window_size=5):
        self.tools = []
        self.llm_pool = None
        self.graph = None
        self.ws = window_size
        self.summarizer = None

    def initialize(self):
        self.tools = [search_google_images, search_web_serper, search_jossa,
                      retrieve_college_allocations_JEE_Main, placement_data,
                      retrieve_college_allocations_JEE_Adv]
        keys = [k.strip() for k in PASTE_GROQ_KEYS_HERE.split(",") if k.strip()]
        if not keys:
            raise ValueError("No keys found please check the api store.")
        llm_instances = []
        for key in keys:
            instance = ChatGroq(
                model="openai/gpt-oss-120b",   # NOTE: Groq expects a valid model name like "mixtral-8x7b-32768"
                temperature=0,
                groq_api_key=key
            ).bind_tools(self.tools)
            llm_instances.append(instance)
        self.llm_pool = itertools.cycle(llm_instances)
        try:
            self.summarizer = ChatGroq(model="llama3-70b-8192", temperature=0, groq_api_key=keys[0])
        except Exception as e:
            print(f"Summarizer init failed: {e}. Using main LLM.")
            self.summarizer = ChatGroq(model="openai/gpt-oss-120b", temperature=0, groq_api_key=keys[0])
        print(f"Successfully initialized LLM Pool with {len(llm_instances)} distinct API connections.")
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("load_memory", self._load_memory_node)
        builder.add_node("agent", self._agent_node)
        builder.add_node("tools", self._tools_node)
        builder.add_node("save_memory", self.save_memory_node)
        builder.add_edge(START, "load_memory")
        builder.add_edge("load_memory", "agent")
        builder.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {"tools": "tools", "save_memory": "save_memory"}
        )
        builder.add_edge("tools", "agent")
        builder.add_edge("save_memory", END)
        return builder.compile()

    async def _load_memory_node(self, state: AgentState) -> dict:
        user_profile = None
        pool = get_pool()
        if pool is not None:
            async with pool.acquire() as conn:
                user_profile = await get_by_id(cast(asyncpg.Connection, conn), state["user_id"])
        if user_profile:
            user_info = f"""
- Name: {user_profile.name}
- Advanced Rank: {user_profile.adv_rank}
- Mains Rank : {user_profile.mains_rank}
- Category: {user_profile.category.value}
- Gender: {user_profile.gender.value}
- Preferred Branches: {', '.join(user_profile.preferred_branches) if user_profile.preferred_branches else 'None'}
"""
        else:
            user_info = "\n- User not found. Please register first."

        system_prompt = f"""
USER PROFILE = {user_info}
*STRICT RULES – YOU MUST FOLLOW THESE*:
1. *College Predictions (JEE Advanced)*:
   - When the user asks for college predictions *for themselves* (e.g., "my rank", "colleges for me"), you MUST call retrieve_college_allocations_JEE_Adv using the *user's own Advanced Rank*, Category, and Gender from the USER PROFILE above.
   - DO NOT ask the user for their rank again – it is already provided in the profile.
   - Example: if the profile shows Advanced Rank: 1500, Category: OPEN, Gender: Gender-Neutral, call the tool with those exact values.
   - NEVER answer from your own knowledge. The tool returns the only correct data.

2. *College Predictions (JEE Main)*:
   - Similar rule – use the user's mains_rank if available, otherwise inform the user.

3. *Other tools* (placement, images, rules, web search) – use as before.

4. rely on tools data for facts for answer formation do not make facts of your own.

5. if user asks query related to specfic 2-3 colleges then return images if important.

*STRICT FORMATTING RULES*:
- When a tool returns college allocation data (a list of records), you MUST render it as a markdown table. NEVER dump raw JSON or Python dicts.
- The markdown table must have these exact columns: | Institute | Academic Program | Opening Rank | Closing Rank | Allotted On |
- Every row from the tool result must appear as a table row. Do not skip or summarize rows.
- Example table format:
| Institute | Academic Program | Opening Rank | Closing Rank | Allotted On |
|-----------|-----------------|--------------|--------------|-------------|
| IIT Bombay | Computer Science and Engineering (4 Years B.Tech) | 1 | 66 | JEE Advanced |
- Before the table, add one line: "College options for your JEE Advanced rank (rank, category, gender):"
- Keep all other answers concise prose.
- search_josaa outputs are authoritative do not ignore them.and give query for 2025 in josaa tool .
"""
        convmsg = [SystemMessage(content=system_prompt)]
        if state.get("summary"):
            convmsg.append(SystemMessage(content=f"Previous conversation summary:\n{state['summary']}"))
        for item in state.get("short_term_memory", [])[-10:]:
            if item.get("role") == "user":
                convmsg.append(HumanMessage(content=item["content"]))
            elif item.get("role") == "ai":
                convmsg.append(AIMessage(content=item["content"]))
        convmsg.append(HumanMessage(content=state.get("current_input")))
        return {"messages": convmsg}

    async def _agent_node(self, state: AgentState) -> dict:
        current_llm = next(self.llm_pool)
        try:
            response = await current_llm.ainvoke(state["messages"])
            return {"messages": [response]}
        except Exception as e:
            print(f"   [LLM Pool Failover Triggered] Error context: {e}")
            fallback = AIMessage(content="⚠️ Request peak limit reached across current pipeline node. Retrying connection...")
            return {"messages": [fallback]}

    async def _tools_node(self, state: AgentState) -> dict:
        last_message = state["messages"][-1]
        tools_lookup = {t.name: t for t in self.tools}
        tool_results = []
        for tool_call in last_message.tool_calls:
            tool_func = tools_lookup.get(tool_call["name"])
            if tool_func:
                result = await tool_func.ainvoke(tool_call["args"])
                tool_results.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        return {"messages": tool_results}

    def _route_after_agent(self, state: AgentState) -> Literal["tools", "save_memory"]:
        last_message = state["messages"][-1]
        return "tools" if hasattr(last_message, "tool_calls") and last_message.tool_calls else "save_memory"

    async def save_memory_node(self, state: AgentState) -> dict:
        print("\n[SAVE MEMORY NODE]")
        final_response = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                final_response = msg.content
                break
        current_mem = state.get("short_term_memory", [])
        current_mem.append({"role": "user", "content": state["current_input"]})
        current_mem.append({"role": "ai", "content": final_response})
        max_messages = 12
        old_summary = state.get("summary", "")
        new_summary = old_summary
        if len(current_mem) > max_messages:
            keep = len(current_mem) // 2
            deleted = current_mem[:-keep]
            remaining = current_mem[-keep:]
            new_summary = await self.summarize_messages(deleted, old_summary)
            current_mem = remaining
            print(f"   Trimmed memory, kept {len(remaining)} messages.")
        else:
            new_summary = old_summary
        pool = get_pool()
        if pool:
            async with pool.acquire() as conn:
                user = await get_by_id(cast(asyncpg.Connection, conn), state["user_id"])
                if user:
                    new_usage = usage_schema(
                        queries_today=user.usage.queries_today + 1,
                        cooldown_until=user.usage.cooldown_until,
                        last_query=datetime.now()
                    )
                    update_data = upadate_schema(usage=new_usage)
                    await update_user(cast(asyncpg.Connection, conn), user.email, update_data)
        return {
            "short_term_memory": current_mem,
            "summary": new_summary,
            "final_response": final_response
        }

    async def summarize_messages(self, messages: list[dict], existing_summary: str) -> str:
        if not messages:
            return existing_summary
        conv = ""
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            conv += f"{role}: {msg['content']}\n"
        prompt = f"""Previous summary: {existing_summary if existing_summary else 'None'}

New conversation excerpt:
{conv}

Produce a concise summary (max 200 tokens) integrating old and new."""
        try:
            response = await self.summarizer.ainvoke([
                SystemMessage(content="You are a summarisation assistant."),
                HumanMessage(content=prompt)
            ])
            content = response.content
            if isinstance(content, list):
                content = ' '.join(str(part) for part in content)
            return content.strip()
        except Exception as e:
            print(f"Summarization failed: {e}")
            return existing_summary

    async def chat(self, user_message: str, user_id: int, session_id: str,
                   short_term_memory: list, summary: str) -> dict:
        initial_state = {
            "messages": [],
            "short_term_memory": short_term_memory,
            "session_id": session_id,
            "user_id": user_id,
            "current_input": user_message,
            "summary": summary
        }
        final_state = await self.graph.ainvoke(cast(AgentState, initial_state))
        return {
            "updated_memory": final_state.get("short_term_memory", []),
            "new_summary": final_state.get("summary", ""),
            "final_response": final_state.get("final_response", "")
        }

    def get_stream(self, initial_state: dict):
        return self.graph.astream(initial_state, stream_mode="messages")

agent = OrchestratorAgent(window_size=5)

class ChatRequest(BaseModel):
    query: str = ""
    email: str
    session_id: Optional[str] = "default_session"
    name: Optional[str] = None
    adv_rank: Optional[int] = None
    mains_rank: Optional[int] = None
    category: Optional[str] = "OPEN"
    gender: Optional[str] = "Gender-Neutral"

class CheckUserRequest(BaseModel):
    email: str

@app.post("/check-user")
async def check_user(request: CheckUserRequest):
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not ready")
    async with pool.acquire() as conn:
        user = await get_by_email(conn, request.email)
        return {"exists": user is not None}

@app.post("/chat")
async def joshai(request: ChatRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not ready")

    async with pool.acquire() as conn:
        user = await get_by_email(conn, request.email)
        if user is None:
            usage = usage_schema(queries_today=0)
            try:
                cat_enum = Category(request.category)
            except ValueError:
                cat_enum = Category.OPEN
            try:
                gen_enum = Gender(request.gender)
            except ValueError:
                gen_enum = Gender.male
            user_data = create_schema(
                name=request.name or request.email.split("@")[0],
                email=request.email,
                adv_rank=request.adv_rank or 0,
                mains_rank=request.mains_rank,
                category=cat_enum,
                gender=gen_enum,
                preferred_branches=[],
                usage=usage,
                short_term_memory=[],
                summary=""
            )
            user = await create_user(conn, user_data)
            print(f"Created new user: {user.name} (id={user.id})")
        else:
            print(f"Existing user: {user.name} (id={user.id})")
        user_id = user.id
        short_term_memory = user.short_term_memory or []
        summary = user.summary or ""

    initial_state = {
        "messages": [],
        "short_term_memory": short_term_memory,
        "session_id": request.session_id,
        "user_id": user_id,
        "current_input": request.query,
        "summary": summary
    }

    async def token_streamer():
        async for message, metadata in agent.get_stream(initial_state):
            if isinstance(message, AIMessage) and metadata.get("langgraph_node") == "agent":
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tc in message.tool_calls:
                        yield f"||TOOL_CALL:{tc['name']}||"
                        
                content = message.content
                if isinstance(content, list):
                    content = ' '.join(str(part) for part in content)
                if content:
                    yield content

    return StreamingResponse(token_streamer(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)