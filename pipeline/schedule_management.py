"""Create one temporary schedule, find its jobs, and disable it."""
from datetime import datetime, timedelta, timezone
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.ml.entities import JobSchedule, RecurrenceTrigger


def get_schedule_or_none(ml_client, schedule_name):
    try:
        return ml_client.schedules.get(schedule_name)
    except ResourceNotFoundError:
        return None


def create_temporary_schedule(ml_client, schedule_name,
                              pipeline_job_definition):
    start = datetime.now(timezone.utc) + timedelta(minutes=5)
    trigger = RecurrenceTrigger(
        frequency="day",
        interval=1,
        start_time=start,
        time_zone="UTC",
    )
    schedule = JobSchedule(
        name=schedule_name,
        display_name="Udacity Project 2 temporary evidence schedule",
        description="Temporary evidence schedule; disable after one run.",
        trigger=trigger,
        create_job=pipeline_job_definition,
        tags={"project": "udacity-project-2", "purpose": "evidence"},
    )
    return ml_client.schedules.begin_create_or_update(schedule).result()


def find_completed_scheduled_job(ml_client, schedule_name):
    matches = []
    for item in ml_client.jobs.list():
        job = ml_client.jobs.get(item.name)
        blob = (
            str(getattr(job, "properties", {}) or {})
            + str(getattr(job, "tags", {}) or {})
        ).lower()
        if job.status == "Completed" and schedule_name.lower() in blob:
            matches.append(job)
    matches.sort(
        key=lambda x: x.creation_context.created_at if x.creation_context else "",
        reverse=True,
    )
    return matches[0] if matches else None


def enable_schedule(ml_client, schedule_name):
    ml_client.schedules.begin_enable(schedule_name).result()
    return ml_client.schedules.get(schedule_name)


def disable_schedule(ml_client, schedule_name):
    schedule = get_schedule_or_none(ml_client, schedule_name)
    if schedule is None:
        return None
    if schedule.is_enabled:
        ml_client.schedules.begin_disable(schedule_name).result()
    return ml_client.schedules.get(schedule_name)


if __name__ == "__main__":
    print("Import this module from Project_2_Master_Notebook_SDK_v2_Working.ipynb.")