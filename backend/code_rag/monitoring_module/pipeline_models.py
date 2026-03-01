from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from jenkins_client import JenkinsClient


class PipelineStage(Enum):
    BUILD = "build"
    DEPLOY = "deploy"
    UNKNOWN = "unknown"


@dataclass
class PipelineResult:
    job_name: str
    build_number: int
    status: str                 # SUCCESS / FAILURE / ABORTED
    stage: PipelineStage
    started_at: datetime
    duration_ms: int
    log: str

def detect_stage_from_log(log: str) -> PipelineStage:
    log_lower = log.lower()

    if "mvn" in log_lower or "gradle" in log_lower or "npm run build" in log_lower:
        return PipelineStage.BUILD

    if "kubectl" in log_lower or "helm" in log_lower or "deploy" in log_lower:
        return PipelineStage.DEPLOY

    return PipelineStage.UNKNOWN


class CICDErrorHandler:
    def __init__(self, jenkins_client: JenkinsClient):
        self.jenkins = jenkins_client

    def process_job_result(self, job_name: str, build_number: int) -> Optional[PipelineResult]:
        build_info = self.jenkins.get_build_info(job_name, build_number)

        status = build_info.get("result")
        if status == "SUCCESS":
            return None  # Ошибки нет — ничего не делаем

        log = self.jenkins.get_build_log(job_name, build_number)

        stage = detect_stage_from_log(log)

        return PipelineResult(
            job_name=job_name,
            build_number=build_number,
            status=status,
            stage=stage,
            started_at=datetime.fromtimestamp(build_info["timestamp"] / 1000),
            duration_ms=build_info["duration"],
            log=log
        )


class LLMAnalyzer:
    def analyze(self, pipeline_result: PipelineResult) -> Dict:
        """
        Здесь вызывается корпоративная LLM:
        - анализ логов
        - классификация ошибки
        - предложение решения
        """

        # Пример результата
        return {
            "error_category": "BUILD_DEPENDENCY_ERROR",
            "root_cause": "Не найдена версия зависимости",
            "suggested_fix": "Проверь версию пакета в pom.xml",
            "responsible_role": "developer"
        }


class NotificationService:
    def notify(self, recipient: str, message: str):
        # Slack / MS Teams / Mattermost / Telegram
        print(f"[NOTIFY] {recipient}: {message}")


def handle_jenkins_failure(
    job_name: str,
    build_number: int,
    jenkins_client: JenkinsClient,
    llm: LLMAnalyzer,
    notifier: NotificationService
):
    handler = CICDErrorHandler(jenkins_client)
    result = handler.process_job_result(job_name, build_number)

    if not result:
        return

    analysis = llm.analyze(result)

    message = (
        f"❌ CI/CD ошибка\n"
        f"Job: {job_name} #{build_number}\n"
        f"Этап: {result.stage.value}\n"
        f"Причина: {analysis['root_cause']}\n"
        f"Решение: {analysis['suggested_fix']}"
    )

    notifier.notify(
        recipient=analysis["responsible_role"],
        message=message
    )
