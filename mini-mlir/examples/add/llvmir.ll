; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

define { ptr, ptr, i64, [4 x i64], [4 x i64] } @main(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr %11, ptr %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21) {
  %23 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  %24 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  %25 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %26 = ptrtoint ptr %25 to i64
  %27 = add i64 %26, 63
  %28 = urem i64 %27, 64
  %29 = sub i64 %27, %28
  %30 = inttoptr i64 %29 to ptr
  %31 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %25, 0
  %32 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %31, ptr %30, 1
  %33 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %32, i64 0, 2
  %34 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %33, i64 1, 3, 0
  %35 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %34, i64 2, 3, 1
  %36 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %35, i64 72, 3, 2
  %37 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %36, i64 2, 3, 3
  %38 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %37, i64 288, 4, 0
  %39 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %38, i64 144, 4, 1
  %40 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %39, i64 2, 4, 2
  %41 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %40, i64 1, 4, 3
  br label %42

42:                                               ; preds = %97, %22
  %43 = phi i64 [ %98, %97 ], [ 0, %22 ]
  %44 = icmp slt i64 %43, 1
  br i1 %44, label %45, label %99

45:                                               ; preds = %42
  br label %46

46:                                               ; preds = %95, %45
  %47 = phi i64 [ %96, %95 ], [ 0, %45 ]
  %48 = icmp slt i64 %47, 2
  br i1 %48, label %49, label %97

49:                                               ; preds = %46
  br label %50

50:                                               ; preds = %93, %49
  %51 = phi i64 [ %94, %93 ], [ 0, %49 ]
  %52 = icmp slt i64 %51, 72
  br i1 %52, label %53, label %95

53:                                               ; preds = %50
  br label %54

54:                                               ; preds = %57, %53
  %55 = phi i64 [ %92, %57 ], [ 0, %53 ]
  %56 = icmp slt i64 %55, 2
  br i1 %56, label %57, label %93

57:                                               ; preds = %54
  %58 = mul i64 %55, 144
  %59 = add i64 0, %58
  %60 = mul i64 %47, 72
  %61 = add i64 %59, %60
  %62 = add i64 %61, %51
  %63 = getelementptr float, ptr %1, i64 %62
  %64 = load float, ptr %63, align 4
  %65 = getelementptr float, ptr %23, i64 0
  store float %64, ptr %65, align 4
  %66 = mul i64 %55, 144
  %67 = add i64 0, %66
  %68 = mul i64 %47, 72
  %69 = add i64 %67, %68
  %70 = add i64 %69, %51
  %71 = getelementptr float, ptr %12, i64 %70
  %72 = load float, ptr %71, align 4
  %73 = getelementptr float, ptr %24, i64 0
  store float %72, ptr %73, align 4
  %74 = add i64 %43, 0
  %75 = add i64 %74, 0
  %76 = add i64 %75, 0
  %77 = getelementptr float, ptr %23, i64 %76
  %78 = load float, ptr %77, align 4
  %79 = add i64 %43, 0
  %80 = add i64 %79, 0
  %81 = add i64 %80, 0
  %82 = getelementptr float, ptr %24, i64 %81
  %83 = load float, ptr %82, align 4
  %84 = fadd float %78, %83
  %85 = mul i64 %43, 288
  %86 = mul i64 %47, 144
  %87 = add i64 %85, %86
  %88 = mul i64 %51, 2
  %89 = add i64 %87, %88
  %90 = add i64 %89, %55
  %91 = getelementptr float, ptr %30, i64 %90
  store float %84, ptr %91, align 4
  %92 = add i64 %55, 1
  br label %54

93:                                               ; preds = %54
  %94 = add i64 %51, 1
  br label %50

95:                                               ; preds = %50
  %96 = add i64 %47, 1
  br label %46

97:                                               ; preds = %46
  %98 = add i64 %43, 1
  br label %42

99:                                               ; preds = %42
  ret { ptr, ptr, i64, [4 x i64], [4 x i64] } %41
}

define void @_mlir_ciface_main(ptr %0, ptr %1, ptr %2) {
  %4 = load { ptr, ptr, i64, [4 x i64], [4 x i64] }, ptr %1, align 8
  %5 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %4, 0
  %6 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %4, 1
  %7 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %4, 2
  %8 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %4, 3, 0
  %9 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %4, 3, 1
  %10 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %4, 3, 2
  %11 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %4, 3, 3
  %12 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %4, 4, 0
  %13 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %4, 4, 1
  %14 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %4, 4, 2
  %15 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %4, 4, 3
  %16 = load { ptr, ptr, i64, [4 x i64], [4 x i64] }, ptr %2, align 8
  %17 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %16, 0
  %18 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %16, 1
  %19 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %16, 2
  %20 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %16, 3, 0
  %21 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %16, 3, 1
  %22 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %16, 3, 2
  %23 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %16, 3, 3
  %24 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %16, 4, 0
  %25 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %16, 4, 1
  %26 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %16, 4, 2
  %27 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %16, 4, 3
  %28 = call { ptr, ptr, i64, [4 x i64], [4 x i64] } @main(ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, ptr %17, ptr %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27)
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %28, ptr %0, align 8
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
