from sqlalchemy.orm import Session

from backend.models.user import User


# 根据用户ID查询单个用户
def get_user(
    db:Session,
    user_id:int
):

    user=db.query(User)\
        .filter(User.id==user_id)\
        .first()

    return user


# 查询所有用户
def get_users(
    db:Session
):

    users=db.query(User).all()

    return users


# 创建新的用户记录
def create_user(
    db:Session,
    username:str,
    email:str
):

    new_user=User(
        username=username,
        email=email
    )


    db.add(new_user)

    db.commit()

    db.refresh(new_user)


    return new_user


# 更新用户信息并返回最新结果
def update_user(
    db:Session,
    user_id:int,
    username:str=None,
    email:str=None
):

    db_user=db.query(User)\
        .filter(User.id==user_id)\
        .first()


    if db_user is None:
        return None


    if username is not None:
        db_user.username=username


    if email is not None:
        db_user.email=email


    db.commit()

    db.refresh(db_user)


    return db_user


# 删除指定用户
def delete_user(
    db:Session,
    user_id:int
):

    user=db.query(User)\
        .filter(User.id==user_id)\
        .first()


    if user is None:
        return False


    db.delete(user)

    db.commit()


    return True