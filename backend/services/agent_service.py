from sqlalchemy.orm import Session

from backend.models.agent import Agent
from backend.models.model import Model
from backend.schemas.agent import AgentCreate, AgentUpdate


class AgentService:

    # 创建新的智能体并保存到数据库
    def create_agent(
        self,
        db: Session,
        agent_data: AgentCreate,
        user_id: int
    ):

        # 1.检查模型是否存在
        model = (
            db.query(Model)
            .filter(Model.id == agent_data.model_id)
            .first()
        )

        if not model:
            raise Exception(
                "Model not found"
            )

        # 2.创建Agent对象
        agent = Agent(
            user_id=user_id,
            name=agent_data.name,
            system_prompt=agent_data.system_prompt,
            model_id=agent_data.model_id
        )

        # 3.保存
        db.add(agent)

        db.commit()

        db.refresh(agent)

        return agent

    # 返回指定用户下的所有智能体列表
    def get_agents(self, db: Session, user_id: int):
        agents = db.query(Agent).filter(Agent.user_id == user_id).all()
        return agents

    # 根据智能体ID查询单个智能体
    def get_agent(self, db: Session, agent_id: int):
        return db.query(Agent).filter(Agent.id == agent_id).first()

    # 更新智能体的可变字段并返回更新后的对象
    def update_agent(self, db: Session, agent_id: int, data: AgentUpdate):
        db_agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if db_agent is None:
            return None

        if data.name is not None:
            db_agent.name = data.name
        if data.system_prompt is not None:
            db_agent.system_prompt = data.system_prompt
        if data.model_id is not None:
            # 检查目标 model 是否存在
            model = db.query(Model).filter(Model.id == data.model_id).first()
            if not model:
                # model 不存在，返回 None 表示失败（router 会抛 404）
                return None
            db_agent.model_id = data.model_id

        db.commit()
        db.refresh(db_agent)
        return db_agent

    # 删除指定智能体
    def delete_agent(self, db: Session, agent_id: int):
        db_agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if db_agent is None:
            return False
        db.delete(db_agent)
        db.commit()
        return True