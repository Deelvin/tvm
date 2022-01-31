#ifndef TVM_RELAY_QNN_OP_DEQUANTIZE_H_
#define TVM_RELAY_QNN_OP_DEQUANTIZE_H_

#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>
namespace tvm {
namespace relay {
namespace qnn {

Expr MakeDequantize(Expr data, Expr input_scale, Expr input_zero_point, int axis);

Expr DequantizeLower(const Expr& input_tensor, const Expr& input_scale,
                     const Expr& input_zero_point, const Array<tvm::relay::Type>& types,
                     const DequantizeAttrs* attrs);


}  // namespace qnn
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_QNN_OP_DEQUANTIZE_H_