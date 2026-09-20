const WORKSPACE_INSTANCE_ID_PATTERN = /^[a-f0-9]{64}$/;

export function resolveAttachedWorkspaceInstanceId(
  value: string | undefined,
  hasWorkspace: boolean
): string | undefined {
  const workspaceInstanceId = value?.trim();
  if (workspaceInstanceId == null || workspaceInstanceId === '') {
    return undefined;
  }
  if (!hasWorkspace || !WORKSPACE_INSTANCE_ID_PATTERN.test(workspaceInstanceId)) {
    throw new Error('Invalid attached workspace instance identifier');
  }
  return workspaceInstanceId;
}
