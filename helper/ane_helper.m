// ane_helper.m - Minimal ANE bridge for tinygrad
// Compile: clang -framework Foundation -framework AppleNeuralEngine -o ane_helper ane_helper.m
// Sign: codesign -s "Developer ID" --entitlements ane_helper.entitlements ane_helper
//
// Protocol: JSON over stdin/stdout
// Request:  {"cmd": "compile|load|eval|unload", "model_path": "...", ...}
// Response: {"ok": true|false, "error": "...", "data": {...}}

#import <Foundation/Foundation.h>
#import <dlfcn.h>

// Forward declarations for private ANE classes
@interface _ANEClient : NSObject
+ (instancetype)alloc;
- (instancetype)initWithRestrictedAccessAllowed:(BOOL)allowed;
- (BOOL)compileModel:(id)model options:(id)options qos:(int)qos error:(NSError **)error;
- (BOOL)loadModel:(id)model options:(id)options qos:(int)qos error:(NSError **)error;
- (BOOL)unloadModel:(id)model options:(id)options qos:(int)qos error:(NSError **)error;
- (BOOL)evaluateWithModel:(id)model options:(id)options request:(id)request qos:(int)qos error:(NSError **)error;
- (id)conn;
@end

@interface _ANEModel : NSObject
+ (instancetype)alloc;
- (instancetype)initWithModelAtURL:(NSURL *)url 
                               key:(NSString *)key 
                  identifierSource:(int)source 
              cacheURLIdentifier:(NSString *)cacheId 
                 modelAttributes:(id)attrs 
                  standardizeURL:(BOOL)standardize;
- (int)state;
- (NSString *)UUID;
- (uint64_t)programHandle;
@end

@interface _ANERequest : NSObject
+ (instancetype)alloc;
- (instancetype)initWithInputs:(NSArray *)inputs 
                  inputIndices:(NSArray *)inputIndices 
                       outputs:(NSArray *)outputs 
                 outputIndices:(NSArray *)outputIndices 
                 weightsBuffer:(id)weights 
                     perfStats:(id)stats 
                procedureIndex:(int)procIdx 
                  sharedEvents:(id)events 
             transactionHandle:(uint64_t)handle;
@end

// Global client
static _ANEClient *g_client = nil;
static NSMutableDictionary *g_models = nil;  // model_id -> _ANEModel

void init_client(void) {
    if (!g_client) {
        // Load the framework
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        
        Class ANEClient = NSClassFromString(@"_ANEClient");
        g_client = [[ANEClient alloc] initWithRestrictedAccessAllowed:YES];
        g_models = [NSMutableDictionary new];
    }
}

NSDictionary *handle_compile(NSDictionary *request) {
    NSString *modelPath = request[@"model_path"];
    NSString *modelId = request[@"model_id"] ?: [[NSUUID UUID] UUIDString];
    
    if (!modelPath) {
        return @{@"ok": @NO, @"error": @"missing model_path"};
    }
    
    NSURL *url = [NSURL fileURLWithPath:modelPath];
    Class ANEModel = NSClassFromString(@"_ANEModel");
    _ANEModel *model = [[ANEModel alloc] initWithModelAtURL:url
                                                       key:modelId
                                          identifierSource:0
                                      cacheURLIdentifier:nil
                                         modelAttributes:nil
                                          standardizeURL:YES];
    
    if (!model) {
        return @{@"ok": @NO, @"error": @"failed to create model"};
    }
    
    NSError *error = nil;
    BOOL result = [g_client compileModel:model options:nil qos:0 error:&error];
    
    if (!result) {
        return @{@"ok": @NO, @"error": error ? error.localizedDescription : @"compile failed"};
    }
    
    g_models[modelId] = model;
    return @{@"ok": @YES, @"model_id": modelId, @"state": @(model.state)};
}

NSDictionary *handle_load(NSDictionary *request) {
    NSString *modelId = request[@"model_id"];
    _ANEModel *model = g_models[modelId];
    
    if (!model) {
        return @{@"ok": @NO, @"error": @"model not found"};
    }
    
    NSError *error = nil;
    BOOL result = [g_client loadModel:model options:nil qos:0 error:&error];
    
    if (!result) {
        return @{@"ok": @NO, @"error": error ? error.localizedDescription : @"load failed"};
    }
    
    return @{@"ok": @YES, @"model_id": modelId, @"program_handle": @(model.programHandle)};
}

NSDictionary *handle_unload(NSDictionary *request) {
    NSString *modelId = request[@"model_id"];
    _ANEModel *model = g_models[modelId];
    
    if (!model) {
        return @{@"ok": @NO, @"error": @"model not found"};
    }
    
    NSError *error = nil;
    BOOL result = [g_client unloadModel:model options:nil qos:0 error:&error];
    
    [g_models removeObjectForKey:modelId];
    
    if (!result) {
        return @{@"ok": @NO, @"error": error ? error.localizedDescription : @"unload failed"};
    }
    
    return @{@"ok": @YES};
}

NSDictionary *handle_status(NSDictionary *request) {
    return @{
        @"ok": @YES,
        @"client": g_client ? @YES : @NO,
        @"model_count": @(g_models.count),
        @"model_ids": g_models.allKeys
    };
}

NSDictionary *handle_request(NSDictionary *request) {
    NSString *cmd = request[@"cmd"];
    
    if ([cmd isEqualToString:@"compile"]) {
        return handle_compile(request);
    } else if ([cmd isEqualToString:@"load"]) {
        return handle_load(request);
    } else if ([cmd isEqualToString:@"unload"]) {
        return handle_unload(request);
    } else if ([cmd isEqualToString:@"status"]) {
        return handle_status(request);
    } else {
        return @{@"ok": @NO, @"error": [NSString stringWithFormat:@"unknown cmd: %@", cmd]};
    }
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        init_client();
        
        // Check if client was created
        if (!g_client) {
            NSDictionary *error = @{@"ok": @NO, @"error": @"failed to create ANE client"};
            NSData *data = [NSJSONSerialization dataWithJSONObject:error options:0 error:nil];
            fwrite(data.bytes, 1, data.length, stdout);
            printf("\n");
            return 1;
        }
        
        // Read JSON from stdin, write JSON to stdout
        // One request per line, one response per line
        char buffer[1024 * 1024];  // 1MB buffer for large requests
        
        while (fgets(buffer, sizeof(buffer), stdin)) {
            NSData *inputData = [NSData dataWithBytes:buffer length:strlen(buffer)];
            NSError *parseError = nil;
            NSDictionary *request = [NSJSONSerialization JSONObjectWithData:inputData options:0 error:&parseError];
            
            NSDictionary *response;
            if (parseError) {
                response = @{@"ok": @NO, @"error": parseError.localizedDescription};
            } else {
                response = handle_request(request);
            }
            
            NSData *outputData = [NSJSONSerialization dataWithJSONObject:response options:0 error:nil];
            fwrite(outputData.bytes, 1, outputData.length, stdout);
            printf("\n");
            fflush(stdout);
        }
    }
    return 0;
}
