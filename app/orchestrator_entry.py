# # app/orchestrator_entry.py
# from app.agents.orchestrator import ExamProcessingOrchestrator

# def process_digital_evaluation(question_path: str, answer_path: str):
#     orchestrator = ExamProcessingOrchestrator()
#     result = orchestrator.run(question_path, answer_path)
#     return result





import asyncio
import os
from app.agents.orchestrator import ExamProcessingOrchestrator

# def process_digital_evaluation(question_path, answer_path, output_folder="output", model="gemini"):
#     """Wrapper to run async orchestration synchronously"""
#     orchestrator = ExamProcessingOrchestrator()
    
#     # Ensure output folder exists
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Run the async process synchronously
#     result = asyncio.run(
#         orchestrator.process_exam_documents(
#             question_pdf=question_path,
#             answer_pdf=answer_path,
#             output_folder=output_folder,
#             selected_model=model
#         )
#     )
#     return result



async def process_digital_evaluation(question_path, answer_path, output_folder="output", model="gemini"):
    """Async wrapper for orchestration"""
    orchestrator = ExamProcessingOrchestrator()
    os.makedirs(output_folder, exist_ok=True)

    result = await orchestrator.process_exam_documents(
        question_pdf=question_path,
        answer_pdf=answer_path,
        output_folder=output_folder,
        selected_model=model
    )
    return result