from datetime import datetime

from sqlalchemy.orm import Session

from backend.models.agent import Agent
# from backend.models.model import Model

from backend.schemas.agent import AgentCreate, AgentUpdate


class AgentService:


    # 创建新的智能体并保存到数据库
    def create_agent(
        self,
        db: Session,
        agent_data: AgentCreate,
        user_id: int
    ):

        # 未来如果恢复Model系统，
        # 可以在这里检查model是否存在

        # 创建Agent对象
        agent = Agent(
            user_id=user_id,
            name=agent_data.name,
            system_prompt=agent_data.system_prompt
        )


        # 保存数据库
        db.add(agent)

        db.commit()

        db.refresh(agent)

        return agent



    # 返回指定用户下的所有智能体列表
    # 只返回未删除的Agent
    def get_agents(
        self,
        db: Session,
        user_id: int
    ):

        agents = (
            db.query(Agent)
            .filter(
                Agent.user_id == user_id,
                Agent.deleted_at == None
            )
            .all()
        )

        return agents



    # 根据智能体ID查询单个智能体
    # 已删除Agent不返回
    def get_agent(
        self,
        db: Session,
        agent_id: int
    ):

        agent = (
            db.query(Agent)
            .filter(
                Agent.id == agent_id,
                Agent.deleted_at == None
            )
            .first()
        )

        return agent



    # 更新智能体信息
    def update_agent(
        self,
        db: Session,
        agent_id: int,
        data: AgentUpdate
    ):

        db_agent = (
            db.query(Agent)
            .filter(
                Agent.id == agent_id,
                Agent.deleted_at == None
            )
            .first()
        )


        if db_agent is None:
            return None


        if data.name is not None:
            db_agent.name = data.name


        if data.system_prompt is not None:
            db_agent.system_prompt = data.system_prompt



        db.commit()

        db.refresh(db_agent)

        return db_agent



    # 删除指定智能体
    # 使用软删除，不真正删除数据库记录
    def delete_agent(
        self,
        db: Session,
        agent_id: int
    ):

        db_agent = (
            db.query(Agent)
            .filter(
                Agent.id == agent_id,
                Agent.deleted_at == None
            )
            .first()
        )


        if db_agent is None:
            return False



        # 软删除
        db_agent.deleted_at = datetime.utcnow()


        db.commit()

        db.refresh(db_agent)


        return True