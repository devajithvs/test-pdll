#include "llvm/Support/SourceMgr.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Parser/Parser.h"

#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/ParseUtilities.h"

#include <iostream>

using namespace mlir;
using namespace llvm;
#include "testPDLL.h.inc"

//===----------------------------------------------------------------------===//
// Query Parser
//===----------------------------------------------------------------------===//

mlir::LogicalResult testPDLLMain(int argc, char **argv, MLIRContext &context) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile("testPDLL.mlir", &errorMessage);
  if (!file) {
    return failure();
  }

  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(file), SMLoc());

  context.allowUnregisteredDialects(true);
  context.printOpOnDiagnostic(false);
  bool noImplicitModule = false;

  // Parse the input MLIR file.
  OwningOpRef<Operation *> opRef =
      parseSourceFileForTool(sourceMgr, &context, !noImplicitModule);
  if (!opRef)
    return failure();

  Operation *rootOp = opRef.get();

  RewritePatternSet patternList(&context);
  populateGeneratedPDLLPatterns(patternList);
  FrozenRewritePatternSet patterns = std::move(patternList);
  (void)applyPatternsAndFoldGreedily(rootOp, patterns);

  rootOp->dump();

  return success();
}

int main() {
  LogicalResult testPDLLMain(int argc, char **argv, MLIRContext &context);
  return 0;
}
