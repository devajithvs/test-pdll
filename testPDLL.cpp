#include "ByteCode.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/ParseUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/SourceMgr.h"
#include <iostream>

#include "testPDLL.h.inc"

using namespace mlir;
using namespace llvm;

//===----------------------------------------------------------------------===//
// Query Parser
//===----------------------------------------------------------------------===//

/// Define a custom PatternRewriter for use by the driver.
class MyPatternRewriter : public PatternRewriter {
public:
  MyPatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}

  /// Override the necessary PatternRewriter hooks here.
};

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

  // rootOp->dump();

  RewritePatternSet patternList(&context);
  populateGeneratedPDLLPatterns(patternList);
  FrozenRewritePatternSet frozenPatternList = std::move(patternList);

  MyPatternRewriter rewriter(&context);

  SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, &context);

  const mlir::detail::PDLByteCode *bytecode =
      frozenPatternList.getPDLByteCode();
  unsigned prevSize = 0;
  unsigned matchCount = 0;
  if (bytecode) {
    auto mutableByteCodeState =
        std::make_unique<mlir::detail::PDLByteCodeMutableState>();
    SmallVector<mlir::detail::PDLByteCode::MatchResult, 4> pdlMatches;
    bytecode->initializeMutableState(*mutableByteCodeState);
    rootOp->walk([&](Operation *op) {
      bytecode->match(op, rewriter, pdlMatches, *mutableByteCodeState);
      matchCount = pdlMatches.size();
      if (matchCount != prevSize) {
        llvm::errs() << "Match #" << matchCount << ":\n\n";
        op->emitRemark("\"root\" binds here");
      }
      prevSize = matchCount;
    });
    llvm::errs() << matchCount
                 << (matchCount == 1 ? " match.\n\n" : " matches.\n\n");
  }

  // (void)applyPatternsAndFoldGreedily(rootOp, patterns);

  // rootOp->dump();

  return success();
}

int main(int argc, char **argv) {

  DialectRegistry registry;
  registerAllDialects(registry);
  MLIRContext context(registry);

  return failed(testPDLLMain(argc, argv, context));
}