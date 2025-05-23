"""
Scheduler.py 

Create / update an EventBridge rule
"""
import boto3, json, os
region = boto3.Session().region_name
events = boto3.client("events", region_name=region)
sm     = boto3.client("sagemaker", region_name=region)

PIPELINE_NAME = "DigitalTwin-Continuous-Training"
RULE_NAME     = "DigitalTwinWeeklyRetrain"


events.put_rule(
    Name=RULE_NAME,
    ScheduleExpression="cron(0 3 ? * SUN *)",  
    State="ENABLED",
    Description=f"Weekly retrain for {PIPELINE_NAME}",
)


events.put_targets(
    Rule=RULE_NAME,
    Targets=[{
        "Id": "TriggerPipeline",
        "Arn": f"arn:aws:sagemaker:{region}:{boto3.client('sts').get_caller_identity()['Account']}:pipeline/{PIPELINE_NAME}",
        "RoleArn": os.environ["EVENTBRIDGE_ROLE"], 
        "Input": json.dumps({"PipelineExecutionStartCondition": "EXPRESSION_MATCH_ONLY"})
    }]
)

print(f"Scheduled {PIPELINE_NAME} via rule {RULE_NAME}")
