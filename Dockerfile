# Use a specialized uv image for faster builds
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy only the files needed for dependency installation to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Install dependencies
# --frozen: ensures uv.lock is not updated
# --no-install-project: avoids installing the current project as a package yet
RUN uv sync --frozen --no-install-project

# Copy the source code and assets
COPY src/ ./src/
COPY README.md ./

# Install the project itself
RUN uv sync --frozen

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables with defaults
ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONPATH=/app/src

# Run the application
CMD ["uv", "run", "-m", "deepseek_client.openai"]
