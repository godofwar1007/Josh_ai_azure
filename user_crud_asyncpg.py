import asyncio
import asyncpg
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List,AsyncGenerator
import enum
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field,ConfigDict
import json


DB_CONFIG = {
    "host": "josh-ai-db.postgres.database.azure.com",
    "port": 5432,
    "database": "orcr_data",
    "user": "postgres",
    "password": "parth@1007",
    "ssl": "require"
}

pool = None

async def init_db_pool():
    global pool
    pool=await asyncpg.create_pool(**DB_CONFIG, min_size=1, max_size=10)

async def close_db_pool():
    global pool
    if pool:
        await pool.close()

async def get_conn():
    async with pool.acquire() as conn:
        yield conn

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("Startup")
#     await init_db_pool()
#     yield
#     print("Shutdown")
#     await close_db_pool()

# app = FastAPI(lifespan=lifespan)

class Gender(str, enum.Enum):
    male = "Male"
    female = "Female"

class Category(str, enum.Enum):
    OPEN = "OPEN"
    EWS = "GEN-EWS"
    OBC_NCL = "OBC-NCL"
    SC = "SC"
    ST = "ST"
    OPEN_PwD = "OPEN(PwD)"
    EWS_PwD = "GEN-EWS(PwD)"
    OBC_NCL_PwD = "OBC-NCL(PwD)"
    SC_PwD = "SC(PwD)"
    ST_PwD = "ST(PwD)"

# @app.get("/test")
# async def test(conn = Depends(get_conn)):
#     result = await conn.fetchval("SELECT 1")
#     return {"result": result}

class usage_schema(BaseModel):
    queries_today:int=0
    cooldown_until:Optional[datetime]=None
    last_query:Optional[datetime]=None

class create_schema(BaseModel):
    name:str
    email:EmailStr
    adv_rank:int
    mains_rank:Optional[int]=None
    category:Category
    gender:Gender
    preferred_branches:List[str]=[]
    usage:usage_schema=Field(default_factory=usage_schema)
    short_term_memory:Optional[List[str]]=None
    summary:Optional[str]=None

class upadate_schema(BaseModel):
    name:Optional[str]=None
    adv_rank:Optional[int]=None
    mains_rank:Optional[int]=None
    category:Optional[Category]=None
    gender:Optional[Gender]=None
    preferred_branches:Optional[List[str]]=None
    usage:Optional[usage_schema]=None
    short_term_memory:Optional[List[str]]=None
    summary:Optional[str]=None

class response_schema(BaseModel):
    id:int
    name:str
    email:str
    adv_rank:int
    mains_rank:Optional[int]
    category:Category
    gender:Gender
    preferred_branches:List[str]
    updated_at:Optional[datetime]
    usage:usage_schema
    short_term_memory:Optional[List[str]]=None
    summary:Optional[str]=None

    model_config = ConfigDict(from_attributes=True)

async def create_user(conn:asyncpg.Connection,user_data:create_schema) -> response_schema:
    usage = user_data.usage
    preferred_branches_json = json.dumps(user_data.preferred_branches)
    short_term_memory_json=json.dumps(user_data.short_term_memory) if user_data.short_term_memory is not None else None
    try:
        row = await conn.fetchrow("""
            INSERT INTO users 
            (name, email, adv_rank, mains_rank, category, gender, preferred_branches,
            queries_today, cooldown_until, last_query,short_term_memory,summary)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,$11::jsonb, $12)
            RETURNING id, name, email, adv_rank, mains_rank, category, gender,
                    preferred_branches, updated_at, queries_today, cooldown_until, last_query,
                    short_term_memory,summary
            """,
            user_data.name,
            user_data.email,
            user_data.adv_rank,
            user_data.mains_rank,
            user_data.category.value,          
            user_data.gender.value,
            preferred_branches_json,
            usage.queries_today,
            usage.cooldown_until,
            usage.last_query,     
            short_term_memory_json,
            user_data.summary
                                
            )
        if row is None:
            raise RuntimeError("failed to insert user: no row returned ")

        preferred_branches_list = json.loads(row["preferred_branches"]) 
        short_term_memory_list=json.loads(row["short_term_memory"]) if row["short_term_memory"] else []         
        result = response_schema(
            id=row["id"],
            name=row["name"],
            email=row["email"],
            adv_rank=row["adv_rank"],
            mains_rank=row["mains_rank"],
            category=Category(row["category"]),
            gender=Gender(row["gender"]),
            preferred_branches=preferred_branches_list,
            updated_at=row["updated_at"],
            usage=usage_schema(
                queries_today=row["queries_today"],
                cooldown_until=row["cooldown_until"],
                last_query=row["last_query"]
            ),
            short_term_memory=short_term_memory_list,
            summary=row["summary"]
            )
        
        return result 
    except Exception as e:
        print(f"Database error {e}")
        raise

async def get_by_email(conn: asyncpg.Connection, email: str) -> Optional[response_schema]:
    row = await conn.fetchrow("SELECT * FROM users WHERE email = $1", email)

    if not row:
        return None
    
    preferred_branches_list = json.loads(row["preferred_branches"])
    short_term_memory_list=json.loads(row["short_term_memory"]) if row["short_term_memory"] else []
    
    result=response_schema(
        id=row["id"],
        name=row["name"],
        email=row["email"],
        adv_rank=row["adv_rank"],
        mains_rank=row["mains_rank"],
        category=Category(row["category"]),
        gender=Gender(row["gender"]),
        preferred_branches=preferred_branches_list,
        updated_at=row["updated_at"],
        usage=usage_schema(
            queries_today=row["queries_today"],
            cooldown_until=row["cooldown_until"],
            last_query=row["last_query"]
        ),
        short_term_memory=short_term_memory_list,
        summary=row["summary"]
    )  

    return result

async def get_by_id(conn: asyncpg.Connection, user_id: int) -> Optional[response_schema]:
    row = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)

    if not row:
        return None
    
    preferred_branches_list = json.loads(row["preferred_branches"])
    short_term_memory_list=json.loads(row["short_term_memory"]) if row["short_term_memory"] else []
    return response_schema(
        id=row["id"],
        name=row["name"],
        email=row["email"],
        adv_rank=row["adv_rank"],
        mains_rank=row["mains_rank"],
        category=Category(row["category"]),
        gender=Gender(row["gender"]),
        preferred_branches=preferred_branches_list,
        updated_at=row["updated_at"],
        usage=usage_schema(
            queries_today=row["queries_today"],
            cooldown_until=row["cooldown_until"],
            last_query=row["last_query"]
        ),
        short_term_memory=short_term_memory_list,
        summary=row["summary"]
    )
async def update_user(conn: asyncpg.Connection, email:str, update_data: upadate_schema) -> Optional[response_schema]:
    
    user=await get_by_email(conn,email)
    if not user:
        return None
    
    user_id=user.id
    sets = []
    params = [] 
    idx = 1
    if update_data.name is not None:
        sets.append(f"name = ${idx}")
        params.append(update_data.name)
        idx += 1
    if update_data.adv_rank is not None:
        sets.append(f"adv_rank = ${idx}")
        params.append(update_data.adv_rank)
        idx += 1
    if update_data.mains_rank is not None:
        sets.append(f"mains_rank = ${idx}")
        params.append(update_data.mains_rank)
        idx += 1
    if update_data.category is not None:
        sets.append(f"category = ${idx}")
        params.append(update_data.category.value)
        idx += 1
    if update_data.gender is not None:
        sets.append(f"gender = ${idx}")
        params.append(update_data.gender.value)
        idx += 1
    if update_data.preferred_branches is not None:
        sets.append(f"preferred_branches = ${idx}")
        params.append(update_data.preferred_branches)
        idx += 1
    if update_data.usage is not None:
        sets.append(f"queries_today = ${idx}")
        params.append(update_data.usage.queries_today)
        idx += 1
        sets.append(f"cooldown_until = ${idx}")
        params.append(update_data.usage.cooldown_until)
        idx += 1
        sets.append(f"last_query = ${idx}")
        params.append(update_data.usage.last_query)
        idx += 1
    if update_data.short_term_memory is not None:
        sets.append(f"short_term_memory = ${idx}::jsonb")
        params.append(json.dumps(update_data.short_term_memory))
        idx += 1   
    if update_data.summary is not None:
        sets.append(f"summary = ${idx}")
        params.append(update_data.summary)
        idx += 1     

    if not sets:
        return await get_by_email(conn, email) # if there is nothing to update just returning user :)

    sets.append("updated_at = NOW()")
    params.append(user_id)

    query = f"""
        UPDATE users
        SET {", ".join(sets)}
        WHERE id = ${idx}
        RETURNING *
    """
    row = await conn.fetchrow(query, *params)

    if not row:
        return None
    
    preferred_branches_list = json.loads(row["preferred_branches"])
    short_term_memory_list=json.loads(row["short_term_memory"]) if row["short_term_memory"] else []
    
    result = response_schema(
        id=row["id"],
        name=row["name"],
        email=row["email"],
        adv_rank=row["adv_rank"],
        mains_rank=row["mains_rank"],
        category=Category(row["category"]),
        gender=Gender(row["gender"]),
        preferred_branches=preferred_branches_list,
        updated_at=row["updated_at"],
        usage=usage_schema(
            queries_today=row["queries_today"],
            cooldown_until=row["cooldown_until"],
            last_query=row["last_query"]
        ),
        short_term_memory=short_term_memory_list,
        summary=row["summary"]
    )

    return result

async def delete_user(conn: asyncpg.Connection, user_id: int) -> bool:
    result = await conn.execute("DELETE FROM users WHERE id = $1", user_id)
    return result.split()[1] != "0"

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db_pool()
    yield
    await close_db_pool()

app = FastAPI(lifespan=lifespan)

@app.post("/users", response_model=response_schema)
async def register_user(user_data:create_schema, conn:asyncpg.Connection=Depends(get_conn)):
    existing=await get_by_email(conn,user_data.email)
    if existing:
        raise HTTPException(400, "Email already registered")
    return await create_user(conn,user_data)

@app.get("/users/{user_id}", response_model=response_schema)
async def fetch_user(user_id:int, conn:asyncpg.Connection=Depends(get_conn)):
    user=await get_by_id(conn, user_id)
    if not user:
        raise HTTPException(404, "User not found")
    return user

@app.patch("/users/{user_id}", response_model=response_schema)
async def modify_user(email:str, update:upadate_schema,conn:asyncpg.Connection=Depends(get_conn)):
    updated=await update_user(conn,email, update)
    if not updated:
        raise HTTPException(404, "User not found")
    return updated

@app.delete("/users/{user_id}")
async def remove_user(user_id:int, conn:asyncpg.Connection=Depends(get_conn)):
    deleted=await delete_user(conn,user_id)
    if not deleted:
        raise HTTPException(404, "User not found")
    return {"message": "User deleted"}

def get_pool():
    return pool
