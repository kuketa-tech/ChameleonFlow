from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Enum as SqlEnum, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from shared.contracts import TransportKind


class Base(DeclarativeBase):
    pass


class SessionAggregateRow(Base):
    __tablename__ = "session_aggregates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    isp_id: Mapped[str] = mapped_column(String(128))
    traffic_type: Mapped[str] = mapped_column(String(64))
    hour_bucket: Mapped[int] = mapped_column(Integer)
    transport: Mapped[TransportKind] = mapped_column(SqlEnum(TransportKind, name="transport_kind"))
    success_count: Mapped[int] = mapped_column(Integer)
    failure_count: Mapped[int] = mapped_column(Integer)
