"""handling table /column overlap

Revision ID: cffefcb2c60a
Revises: 3c8c1dee02c0
Create Date: 2024-05-29 15:00:07.426207

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cffefcb2c60a'
down_revision: Union[str, None] = '3c8c1dee02c0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    pass
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    pass
    # ### end Alembic commands ###
