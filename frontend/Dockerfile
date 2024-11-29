FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY smiles-ui/package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY smiles-ui/ .

# Build the Next.js application with production settings
ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=production
RUN npm run build -- --no-lint

# Expose the port Next.js runs on
EXPOSE 3000

# Start the application
CMD ["npm", "run", "start"]