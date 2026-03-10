from fastapi import APIRouter

router = APIRouter(tags=["fixtures"])


@router.get("/fixtures/upcoming")
def get_upcoming_fixtures():
    # TODO: implement in Step 7
    return {"status": "not implemented yet"}


@router.get("/standings")
def get_standings():
    # TODO: implement in Step 7
    return {"status": "not implemented yet"}
