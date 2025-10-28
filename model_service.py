# # model_service.py
# from fastapi import FastAPI, UploadFile, File
# from app.orchestrator_entry import process_digital_evaluation
# import uvicorn

# app = FastAPI(title="Digital Evaluation Model API")

# @app.post("/process")
# async def process_paper(question_paper: UploadFile = File(...), answer_sheet: UploadFile = File(...)):
#     with open("temp_question.pdf", "wb") as f:
#         f.write(await question_paper.read())
#     with open("temp_answer.pdf", "wb") as f:
#         f.write(await answer_sheet.read())

#     result = process_digital_evaluation("temp_question.pdf", "temp_answer.pdf")

#     return {"status": "success", "data": result}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)





# model_service.py
from fastapi import FastAPI, UploadFile, File
from app.orchestrator_entry import process_digital_evaluation
import uvicorn

app = FastAPI(title="Digital Evaluation Model API")

@app.post("/process")
async def process_paper(
    question_paper: UploadFile = File(...),
    answer_sheet: UploadFile = File(...)
):
    # Save uploaded files temporarily
    with open("temp_question.pdf", "wb") as f:
        f.write(await question_paper.read())
    with open("temp_answer.pdf", "wb") as f:
        f.write(await answer_sheet.read())

    # 👇 Await the async function here
    result = await process_digital_evaluation("temp_question.pdf", "temp_answer.pdf")

    return {"status": "success", "data": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
