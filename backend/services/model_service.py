from sqlalchemy.orm import Session

from backend.models.model import Model
from backend.schemas.model import ModelCreate


class ModelService:

    # 创建模型记录并保存到数据库
    def create_model(
        self,
        db:Session,
        data:ModelCreate
    ):

        model = Model(

            provider=data.provider,

            model_name=data.model_name,

            display_name=data.display_name,

            api_endpoint=data.api_endpoint,

            context_window=data.context_window

        )

        db.add(model)

        db.commit()

        db.refresh(model)

        return model

    # 查询所有模型记录
    def get_models(
        self,
        db:Session
    ):

        return (
            db.query(Model)
            .all()
        )

    # 根据模型ID查询单条模型记录
    def get_model(
        self,
        db:Session,
        model_id:int
    ):

        return (
            db.query(Model)
            .filter(
                Model.id==model_id
            )
            .first()
        )