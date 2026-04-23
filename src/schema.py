from __future__ import annotations
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field

RelationType = Union[str, List[str]]

class Structure(BaseModel):
    doc_title: Optional[str] = None
    nb_paras: Optional[int] = None
    preambular_para: List[int] = Field(default_factory=list)
    operative_para: List[int] = Field(default_factory=list)
    think: str = ""

class Para(BaseModel):
    para_number: int
    para: Optional[str] = None
    para_en: Optional[str] = None
    type: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    matched_pars: Dict[str, RelationType] = Field(default_factory=dict)
    think: str = ""

class Metadata(BaseModel):
    structure: Structure

class Body(BaseModel):
    paragraphs: List[Para]

class Doc(BaseModel):
    TEXT_ID: str
    RECOMMENDATION: Optional[int] = None
    TITLE: Optional[str] = None
    METADATA: Metadata
    body: Body
