"""table adjustment

Revision ID: 5dad00a4a606
Revises: 
Create Date: 2024-05-24 22:19:14.730053

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5dad00a4a606'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('optimizationresult', sa.Column('problem_id', sa.Integer(), nullable=False))
    op.create_foreign_key(None, 'optimizationresult', 'problems', ['problem_id'], ['id'])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'optimizationresult', type_='foreignkey')
    op.drop_column('optimizationresult', 'problem_id')
    # ### end Alembic commands ###