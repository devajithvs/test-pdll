#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/ParseUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "testPDLL.h.inc"
#include "llvm/Support/SourceMgr.h"
#include <iostream>

using namespace mlir;
using namespace llvm;

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

  rootOp->dump();

  RewritePatternSet patternList(&context);
  populateGeneratedPDLLPatterns(patternList);
  FrozenRewritePatternSet patterns = std::move(patternList);
  (void)applyPatternsAndFoldGreedily(rootOp, patterns);

  rootOp->dump();

  return success();
}

int main(int argc, char **argv) {

  DialectRegistry registry;
  registerAllDialects(registry);
  MLIRContext context(registry);

  return failed(testPDLLMain(argc, argv, context));
}