# Etapa de compilación
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json ./
RUN npm install
COPY . .
RUN npm run build -- --configuration production

# Etapa de servidor
FROM nginx:alpine
COPY --from=build /app/dist/previa-frontend/browser /usr/share/nginx/html
# Copia una configuración de Nginx básica para SPA si es necesario
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
