from fastapi import APIRouter

router = APIRouter(tags=["predictions"])


@router.get("/predictions/{match_id}")
def get_prediction(match_id: int):
    # TODO: implement in Step 7
    return {"match_id": match_id, "status": "not implemented yet"}


@router.get("/accuracy")
def get_accuracy():
    # TODO: implement in Step 7
    return {"status": "not implemented yet"}
