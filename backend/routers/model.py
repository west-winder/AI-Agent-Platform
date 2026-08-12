from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database.database import get_db

from fastapi import HTTPException
from backend.schemas.model import (
    ModelCreate,
    ModelResponse
)

from backend.services.model_service import ModelService


router = APIRouter(
    prefix="/models",
    tags=["Model"]
)


model_service = ModelService()


@router.post("", response_model=ModelResponse)
# 创建模型接口
def create_model(
    model: ModelCreate,
    db: Session = Depends(get_db)
):

    return model_service.create_model(
        db,
        model
    )


@router.get("", response_model=list[ModelResponse])
# 获取模型列表接口
def get_models(
    db: Session = Depends(get_db)
):

    return model_service.get_models(
        db
    )


@router.get("/{model_id}", response_model=ModelResponse)
# 获取单个模型接口
def get_model(
    model_id:int,
    db:Session=Depends(get_db)
):

    model = model_service.get_model(
        db,
        model_id
    )


    if not model:

        raise HTTPException(
            status_code=404,
            detail="Model not found"
        )


    return model


@router.delete("/{model_id}")
# 删除模型接口
def delete_model(
    model_id:int,
    db:Session=Depends(get_db)
):

    result=model_service.delete_model(
        db,
        model_id
    )


    if not result:

        raise HTTPException(
            status_code=404
        )


    return {
        "message":"deleted"
    }