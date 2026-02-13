"""
Property-based tests for TrainingJob model.

Uses Hypothesis to generate random inputs and verify properties
that should hold universally across the input space.

Feature: feedback-loop-personalization
Property 16: Training Status Validation

**Validates: Requirements 9.2**
"""

import pytest
from hypothesis import given, strategies as st, settings
from bson import ObjectId

from app.models.training_job import TrainingJob


# ============================================================================
# Property 16: Training Status Validation
# ============================================================================

@given(status=st.text())
@settings(max_examples=200)
def test_property_status_validation(status):
    """
    Property 16: Training Status Validation
    
    For any TrainingJob record, the status field should be one of the valid values:
    "queued", "running", "completed", or "failed", and no other values should be allowed.
    
    This property verifies that:
    1. Valid statuses are accepted without error
    2. Invalid statuses raise ValueError
    3. The error message includes all valid options
    
    **Validates: Requirements 9.2**
    """
    valid_statuses = {"queued", "running", "completed", "failed"}
    is_valid = status in valid_statuses
    
    user_id = str(ObjectId())
    job_id = "test-job-123"
    
    if is_valid:
        # Valid status should be accepted without error
        training_job = TrainingJob(
            user_id=user_id,
            job_id=job_id,
            status=status
        )
        assert training_job.status == status
        assert training_job.status in TrainingJob.VALID_STATUSES
    else:
        # Invalid status should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            TrainingJob(
                user_id=user_id,
                job_id=job_id,
                status=status
            )
        
        # Verify error message format
        error_msg = str(exc_info.value)
        assert "Invalid status" in error_msg
        assert status in error_msg or repr(status) in error_msg
        
        # Verify error message includes all valid options
        for valid_status in valid_statuses:
            assert valid_status in error_msg


@given(
    status=st.sampled_from(["queued", "running", "completed", "failed"])
)
@settings(max_examples=100)
def test_property_valid_statuses_accepted(status):
    """
    Property: All Valid Statuses Accepted
    
    For any status in the valid set {"queued", "running", "completed", "failed"},
    TrainingJob initialization should succeed without error.
    
    **Validates: Requirements 9.2**
    """
    user_id = str(ObjectId())
    job_id = f"job-{status}"
    
    # Should not raise any exception
    training_job = TrainingJob(
        user_id=user_id,
        job_id=job_id,
        status=status
    )
    
    assert training_job.status == status
    assert training_job.status in TrainingJob.VALID_STATUSES


@given(
    status=st.text().filter(lambda s: s not in {"queued", "running", "completed", "failed"})
)
@settings(max_examples=200)
def test_property_invalid_statuses_rejected(status):
    """
    Property: All Invalid Statuses Rejected
    
    For any status NOT in the valid set {"queued", "running", "completed", "failed"},
    TrainingJob initialization should raise ValueError.
    
    This includes:
    - Empty strings
    - Wrong case (e.g., "QUEUED", "Completed")
    - Similar but incorrect values (e.g., "pending", "cancelled")
    - Random strings
    - Special characters
    
    **Validates: Requirements 9.2**
    """
    user_id = str(ObjectId())
    job_id = "job-invalid"
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="Invalid status"):
        TrainingJob(
            user_id=user_id,
            job_id=job_id,
            status=status
        )


@given(
    status=st.sampled_from(["queued", "running", "completed", "failed"])
)
@settings(max_examples=100)
def test_property_status_preserved_in_serialization(status):
    """
    Property: Status Preserved in Serialization
    
    For any valid status, the status value should be preserved through
    serialization (to_dict) and deserialization (from_dict) operations.
    
    **Validates: Requirements 9.2**
    """
    user_id = str(ObjectId())
    job_id = f"job-{status}"
    
    # Create TrainingJob with status
    original = TrainingJob(
        user_id=user_id,
        job_id=job_id,
        status=status
    )
    
    # Serialize to dict
    job_dict = original.to_dict()
    assert job_dict["status"] == status
    
    # Deserialize from dict
    restored = TrainingJob.from_dict(job_dict)
    assert restored.status == status
    assert restored.status == original.status


@given(
    status1=st.sampled_from(["queued", "running", "completed", "failed"]),
    status2=st.sampled_from(["queued", "running", "completed", "failed"])
)
@settings(max_examples=100)
def test_property_status_independence(status1, status2):
    """
    Property: Status Independence
    
    Creating multiple TrainingJob instances with different statuses
    should not affect each other. Each instance maintains its own status.
    
    **Validates: Requirements 9.2**
    """
    user_id = str(ObjectId())
    
    # Create two jobs with potentially different statuses
    job1 = TrainingJob(
        user_id=user_id,
        job_id="job-1",
        status=status1
    )
    
    job2 = TrainingJob(
        user_id=user_id,
        job_id="job-2",
        status=status2
    )
    
    # Verify each maintains its own status
    assert job1.status == status1
    assert job2.status == status2
    
    # Verify they don't interfere with each other
    assert job1.status in TrainingJob.VALID_STATUSES
    assert job2.status in TrainingJob.VALID_STATUSES


@given(
    status=st.sampled_from(["queued", "running", "completed", "failed"])
)
@settings(max_examples=100)
def test_property_status_constant_membership(status):
    """
    Property: Status Constant Membership
    
    For any valid status, it should always be a member of VALID_STATUSES constant.
    The VALID_STATUSES set should contain exactly the four valid status values.
    
    **Validates: Requirements 9.2**
    """
    # Verify status is in VALID_STATUSES
    assert status in TrainingJob.VALID_STATUSES
    
    # Verify VALID_STATUSES contains exactly the expected values
    expected_statuses = {"queued", "running", "completed", "failed"}
    assert TrainingJob.VALID_STATUSES == expected_statuses
    
    # Verify no extra or missing values
    assert len(TrainingJob.VALID_STATUSES) == 4


# ============================================================================
# Additional Property Tests for Status Transitions
# ============================================================================

@given(
    initial_status=st.sampled_from(["queued", "running", "completed", "failed"]),
    new_status=st.sampled_from(["queued", "running", "completed", "failed"])
)
@settings(max_examples=100)
def test_property_status_mutability(initial_status, new_status):
    """
    Property: Status Mutability
    
    A TrainingJob's status can be changed to any other valid status.
    This tests that status is mutable and not frozen after initialization.
    
    Note: This tests the model's capability. Actual status transitions
    should follow business logic rules (e.g., queued -> running -> completed).
    
    **Validates: Requirements 9.2**
    """
    user_id = str(ObjectId())
    job_id = "job-mutable"
    
    # Create job with initial status
    job = TrainingJob(
        user_id=user_id,
        job_id=job_id,
        status=initial_status
    )
    
    assert job.status == initial_status
    
    # Change status (at model level, bypassing business logic)
    job.status = new_status
    
    # Verify status changed
    assert job.status == new_status
    assert job.status in TrainingJob.VALID_STATUSES


@given(
    status=st.sampled_from(["queued", "running", "completed", "failed"])
)
@settings(max_examples=100)
def test_property_status_type_consistency(status):
    """
    Property: Status Type Consistency
    
    For any valid status, the status field should always be a string type.
    
    **Validates: Requirements 9.2**
    """
    user_id = str(ObjectId())
    job_id = f"job-{status}"
    
    job = TrainingJob(
        user_id=user_id,
        job_id=job_id,
        status=status
    )
    
    # Verify status is a string
    assert isinstance(job.status, str)
    
    # Verify it's not some other type that happens to equal the status
    assert type(job.status) == str
