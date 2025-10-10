#!/bin/bash
# Fix gRPC import issues

cd "$(dirname "$0")/network"

echo "🔧 Fixing gRPC imports..."

# Regenerate protobuf files
echo "Regenerating protobuf files..."
../DeepSeek/bin/python -m grpc_tools.protoc --python_out=. --grpc_python_out=. --proto_path=. grpc_service.proto

# Fix the import in the generated grpc file
echo "Fixing imports in generated files..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' 's/import grpc_service_pb2 as grpc__service__pb2/from . import grpc_service_pb2 as grpc__service__pb2/' grpc_service_pb2_grpc.py
else
    # Linux
    sed -i 's/import grpc_service_pb2 as grpc__service__pb2/from . import grpc_service_pb2 as grpc__service__pb2/' grpc_service_pb2_grpc.py
fi

echo "✅ gRPC imports fixed!"
echo ""
echo "Files updated:"
echo "  - grpc_service_pb2.py"
echo "  - grpc_service_pb2_grpc.py"