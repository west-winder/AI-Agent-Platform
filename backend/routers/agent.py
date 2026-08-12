from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session


from backend.database.database import get_db

from backend.schemas.agent import (
    AgentCreate,
    AgentUpdate,
    AgentResponse
)

from backend.services.agent_service import AgentService


router = APIRouter(
    prefix="/agents",
    tags=["Agent"]
)


agent_service = AgentService()


# 创建智能体接口
def create_agent(
    agent: AgentCreate,
    db: Session = Depends(get_db)
):

    user_id = 1

    return agent_service.create_agent(
        db,
        agent,
        user_id
    )


# 获取当前用户下的智能体列表接口
def get_agents(
    db:Session = Depends(get_db)
):

    user_id = 1

    return agent_service.get_agents(
        db,
        user_id
    )


# 获取单个智能体接口
def get_agent(
    agent_id:int,
    db:Session=Depends(get_db)
):

    agent = agent_service.get_agent(
        db,
        agent_id
    )

    if not agent:

        raise HTTPException(
            status_code=404,
            detail="Agent not found"
        )

    return agent


# 更新智能体信息接口
def update_agent(
    agent_id:int,
    data:AgentUpdate,
    db:Session=Depends(get_db)
):

    agent = agent_service.update_agent(
        db,
        agent_id,
        data
    )

    if not agent:

        raise HTTPException(
            status_code=404
        )

    return agent


# 删除智能体接口
def delete_agent(
    agent_id:int,
    db:Session=Depends(get_db)
):

    result = agent_service.delete_agent(
        db,
        agent_id
    )

    if not result:

        raise HTTPException(
            status_code=404
        )

    return {
        "message":"Agent deleted"
    }


