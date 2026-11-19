# Deploying QuerentCrew on AWS

This guide deploys the Docker image to a single **Amazon EC2** instance with Docker Compose. It's the
simplest setup that works on a free-plan account, and it uses services covered by the AWS Certified
Cloud Practitioner syllabus: EC2, security groups, IAM, S3, Systems Manager Parameter Store, CloudWatch
and Budgets.

Expected time: about 45 minutes. Expected cost: ₹0 on a new AWS Free plan account, paid from the free
credits, as long as you use a free-tier instance type and stop it when you're not demoing.

---

## 0. Before you start: protect yourself from bills

1. **Account plan.** Accounts created on or after 15 July 2025 choose a *Free plan* or *Paid plan* at
   sign-up. The Free plan gives $100 in credits (up to $100 more for onboarding tasks) and runs for 6
   months or until credits run out. It never bills you; the account is closed instead.
2. **Create a budget:** Billing and Cost Management → Budgets → *Create budget* → *Zero spend budget*
   (or a monthly cost budget of $5) → add your email.
3. Use only **t2.micro / t3.micro** (marked "Free tier eligible" in the console).
4. **Stop the instance** when you're not using it. A public IPv4 address and EBS storage still cost a
   little while stopped, which comes out of your credits.

---

## 1. Store secrets in Parameter Store (don't keep API keys on disk)

Console → **Systems Manager → Parameter Store → Create parameter**

| Name | Type | Value |
|---|---|---|
| `/querentcrew/GROQ_API_KEY` | SecureString | your Groq key |
| `/querentcrew/LANGSMITH_API_KEY` | SecureString | your LangSmith key (optional) |

## 2. IAM role for the instance

IAM → Roles → *Create role* → trusted entity **EC2** → attach:
- `AmazonSSMManagedInstanceCore` (lets you connect through Session Manager, no SSH key needed)
- an inline policy to read the parameters (and the backup bucket from step 7):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": ["ssm:GetParameter", "ssm:GetParameters"],
      "Resource": "arn:aws:ssm:*:*:parameter/querentcrew/*" },
    { "Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject"],
      "Resource": "arn:aws:s3:::querentcrew-backups-*/*" }
  ]
}
```

Name it `querentcrew-ec2-role`.

## 3. Launch the EC2 instance

EC2 → *Launch instance*:
- **Name:** querentcrew
- **AMI:** Ubuntu Server 24.04 LTS (free tier eligible)
- **Instance type:** t3.micro (or t2.micro, whichever is free-tier eligible in your Region; Mumbai `ap-south-1` is closest)
- **Key pair:** create one (or skip and use Session Manager)
- **Security group:** create `querentcrew-sg` with inbound rules:
  - SSH 22 from **My IP** only
  - Custom TCP **8501** from Anywhere (Streamlit UI)
  - Custom TCP **8000** from My IP (API, optional)
- **Storage:** 20 GB gp3
- **Advanced → IAM instance profile:** `querentcrew-ec2-role`

## 4. Install Docker on the instance

Connect via EC2 Instance Connect, Session Manager or `ssh -i key.pem ubuntu@<public-ip>`:

```bash
# Docker + compose plugin
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2 git awscli
sudo usermod -aG docker ubuntu && newgrp docker

# 1 GB of RAM is tight for installing the Python packages: add 2 GB of swap
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 5. Get the code and create `.env` from Parameter Store

```bash
git clone https://github.com/Himanshu-joshi986/querentcrew.git
cd querentcrew
cp .env.example .env

REGION=ap-south-1   # change to your Region
GROQ=$(aws ssm get-parameter --name /querentcrew/GROQ_API_KEY --with-decryption \
       --query Parameter.Value --output text --region $REGION)
sed -i "s|^GROQ_API_KEY=.*|GROQ_API_KEY=$GROQ|" .env
chmod 600 .env
```

(Optional) to turn on LangSmith tracing, set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` the same way.

## 6. Build and run

```bash
docker compose up -d --build      # first build takes ~5-10 min on a micro instance
docker compose ps
docker compose logs -f ui
```

Open **http://\<public-ip\>:8501** for the UI and **http://\<public-ip\>:8000/docs** for the API (if port 8000 is open to you).

Update after a `git push`:

```bash
git pull && docker compose up -d --build
```

## 7. (Optional) Back up the database and uploaded tables to S3

```bash
aws s3 mb s3://querentcrew-backups-<your-name> --region $REGION
docker compose cp ui:/app/data/sales.db ./sales-backup.db
aws s3 cp ./sales-backup.db s3://querentcrew-backups-<your-name>/sales-$(date +%F).db
```

## 8. (Optional) Logs in CloudWatch

Add this to each service in `docker-compose.yml`:

```yaml
    logging:
      driver: awslogs
      options:
        awslogs-region: ap-south-1
        awslogs-group: /querentcrew
        awslogs-create-group: "true"
```

and add `CloudWatchAgentServerPolicy` to the instance role.

## 9. (Optional) Use Amazon Bedrock as the LLM

LangChain's `init_chat_model` supports Bedrock. Enable a model in the Bedrock console, then:

```bash
pip install langchain-aws          # add to requirements.txt
# .env
LLM_MODEL=bedrock_converse:<model-id-from-the-bedrock-console>
```

Add `bedrock:InvokeModel` to the instance role. Bedrock charges per token and isn't part of the free
allowances, so keep Groq or Gemini as the default for demos.

## 10. Production hardening (what I'd add next)

| Concern | Change |
|---|---|
| HTTPS + custom domain | Put an Application Load Balancer with an ACM certificate (or Nginx + Let's Encrypt) in front |
| Persistent conversation memory | Replace `InMemorySaver` with the LangGraph Postgres checkpointer on RDS |
| Scaling | Push the image to **ECR** and run it on **ECS Fargate** behind the ALB |
| Real data | Point `DATABASE_PATH` at a read replica, or add a Postgres connector, with a read-only DB user |
| Auth | Put Amazon Cognito or the ALB's OIDC auth in front of the UI |

## Clean-up checklist

- [ ] Stop (or terminate) the EC2 instance
- [ ] Release any Elastic IP
- [ ] Delete unused EBS snapshots / volumes
- [ ] Empty and delete the S3 backup bucket if no longer needed
