
import asyncio
import asyncpg

s=[250,120,120,60,30,15]
h=[800,400,400,200,120,40]
low=0
high=0

DB_CONFIG = {
    "host": "josh-ai-db.postgres.database.azure.com",
    "port": 5432,
    "database": "orcr_data",
    "user": "postgres",
    "password": "TempPass123!",   
    "ssl": "require"
}


class ORCR_Retriever:
    async def runa(self,ja_rank,category,gender):#for advanced rank
        conn = await asyncpg.connect(**DB_CONFIG)
        values = await conn.fetch(
            "SELECT * FROM seat_allocation WHERE seat_type =$1 AND gender=$2 AND rank='adv'",category,gender 
        
        )
        

        l=[]
        h2=[]
        c=["OPEN","OBC-NCL","GEN-EWS","SC","ST"]
        if category in c:
            low=s[c.index(category)]
            high=h[c.index(category)]
        else:
            low=25
            high=90
        if ja_rank>=13000:
            low=low*4
            high=high*4
        for row in values:
            r=row["opening_rank"]
            cr=row["closing_rank"]
            institute=row["institute"]
            institute=institute[0:3:1]+" "+ institute[3:len(institute):1]
            record={"Institute":institute,"Academic program":row["academic_program"],"Opening Rank":r,"Closing Rank":cr,"Alloted on basis of":"JEE Advanced" }
            if ja_rank>cr and (ja_rank-cr)<=low:
                l.append(record)
            if ja_rank<cr and (cr-ja_rank)<=high:
                h2.append(record)

        p=sorted(l,key=lambda x:x["Closing Rank"],reverse=False)
        k=sorted(h2,key=lambda x:x["Closing Rank"],reverse=False)
        #print(len(p))
        #print(len(k))
        t=len(p)
        if len(p) >4:
            p=p[-4:-1:1]+[p[-1]]
            t=4
        final=p+k[:(10-t):1]

        await conn.close()
        return final
    
    async def runm(self,jm_rank,category,gender): #for mains rank
        conn = await asyncpg.connect(**DB_CONFIG)
        values = await conn.fetch(
            "SELECT * FROM seat_allocation WHERE seat_type =$1 AND gender=$2 AND rank='mains'",category,gender 
        
        )
        l=[]
        h2=[]
        c=["OPEN","OBC-NCL","GEN-EWS","SC","ST"]
        if category in c:
            low=s[c.index(category)]
            high=h[c.index(category)]
        else:
            low=25
            high=90
        
        low=low*4
        high=high*4

        for row in values:
            r=row["opening_rank"]
            cr=row["closing_rank"]
            institute=row["institute"]
            branch=row["academic_program"]
            quota=row["quota"]
            record={"Institute":institute,"Academic program":row["academic_program"],"Opening Rank":r,"Closing Rank":cr,"Allotted on basis of":"JEE Mains","Quota":quota}
            if jm_rank>cr and (jm_rank-cr)<=low :
                if branch != "Architecture (5 Years, Bachelor of Architecture)" and branch != "Planning (4 Years, Bachelor of Planning)":
                    l.append(record)
            if jm_rank<cr and (cr-jm_rank)<=high:
                if branch != "Architecture (5 Years, Bachelor of Architecture)" and branch != "Planning (4 Years, Bachelor of Planning)":
                    h2.append(record)
        
        p=sorted(l,key=lambda x:x["Closing Rank"],reverse=False)
        
        k=sorted(h2,key=lambda x:x["Closing Rank"],reverse=False)
        
        #print(len(p))
        #print(len(k))
        t=len(p)
        if len(p) >4:
            p=p[-4:-1:1]+[p[-1]]
            t=4
        final=p+k[:(10-t):1]

        await conn.close()
        return final
    
        
        
    
async def main():
    retriever=ORCR_Retriever()
    #options=await retriever.runa(2500,"OPEN","Gender-Neutral")
    options=await retriever.runm(1000,"OPEN","Gender-Neutral")

    print(options)

if __name__== "__main__":
    asyncio.run(main())
