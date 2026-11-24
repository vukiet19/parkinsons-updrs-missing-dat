"""Basic teacher and student; all image encoders are trainable."""

# Both experiments reuse exactly the same prediction architecture.
from common.models import Student, Teacher

__all__ = ['Teacher', 'Student']
