from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.schemas.user import UserCreate, UserUpdate, UserResponse, MessageResponse
from backend.services.user_service import (
   create_user,
   delete_user,
   get_user,
   get_users,
   update_user,
)

router = APIRouter(prefix="/users", tags=["User"])


# 创建用户接口
def create_user_api(
   user: UserCreate,
   db: Session = Depends(get_db)
):
   new_user = create_user(
       db,
       user.username,
       user.email,
   )
   return new_user


# 获取所有用户接口
def get_users_api(
   db: Session = Depends(get_db)
):
   users = get_users(db)
   return users


# 根据用户ID获取用户接口
def get_user_api(
   user_id: int,
   db: Session = Depends(get_db)
):
   user = get_user(
       db,
       user_id,
   )

   if user is None:
       raise HTTPException(
           status_code=404,
           detail="User not found",
       )

   return user


# 更新用户信息接口
def update_user_api(
   user_id: int,
   user: UserUpdate,
   db: Session = Depends(get_db)
):
   db_user = update_user(
       db,
       user_id,
       user.username,
       user.email,
   )

   if db_user is None:
       raise HTTPException(
           status_code=404,
           detail="User not found",
       )

   return db_user


# 删除用户接口
def delete_user_api(
   user_id: int,
   db: Session = Depends(get_db)
):
   result = delete_user(
       db,
       user_id,
   )

   if result is False:
       raise HTTPException(
           status_code=404,
           detail="User not found",
       )

   return {
       "message": "User deleted successfully"
   }