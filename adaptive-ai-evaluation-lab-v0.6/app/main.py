from fastapi import FastAPI,WebSocket,WebSocketDisconnect
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.db import Base,engine,SessionLocal
from app.domain import db_models as m
from app.api.router import router
from app.orchestration.events import bus
@asynccontextmanager
async def lifespan(app):
 Base.metadata.create_all(engine);db=SessionLocal()
 try:
  if settings.auto_seed and not db.query(m.ModelEndpoint).count():db.add_all([m.ModelEndpoint(name="Mock Strong",adapter="mock",model_name="mock-strong"),m.ModelEndpoint(name="Mock Balanced",adapter="mock",model_name="mock-balanced"),m.ModelEndpoint(name="Mock Fragile",adapter="mock",model_name="mock-fragile")]);db.commit()
 finally:db.close()
 yield
app=FastAPI(title=settings.app_name,version="0.6.0",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=settings.cors_list,allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
app.include_router(router)
@app.websocket("/ws/experiments/{eid}")
async def ws(websocket:WebSocket,eid:int):
 await websocket.accept(); q=bus.subscribe(eid)
 try:
  while True: await websocket.send_json(await q.get())
 except WebSocketDisconnect: pass
 finally: bus.unsubscribe(eid,q)
